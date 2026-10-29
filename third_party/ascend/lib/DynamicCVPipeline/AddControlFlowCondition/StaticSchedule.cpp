/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/StaticSchedule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

// Appends the merged [begin, end) runs of `fires` over [from, to).
void appendRuns(llvm::ArrayRef<char> fires, int64_t from, int64_t to,
                llvm::SmallVectorImpl<IvrRange> &ranges) {
  int64_t begin = -1;
  for (int64_t t = from; t < to; ++t) {
    if (fires[t] && begin < 0) {
      begin = t;
    } else if (!fires[t] && begin >= 0) {
      ranges.push_back({begin, t});
      begin = -1;
    }
  }
  if (begin >= 0)
    ranges.push_back({begin, to});
}

// Compacts a firing vector into a prefix range list plus at most one periodic
// tail. The tail search keeps the emitted predicate small for rotation
// schedules, which fire every `period` steps once the prologue is over.
StaticBlockSchedule compactSchedule(llvm::ArrayRef<char> fires) {
  StaticBlockSchedule schedule;
  int64_t size = static_cast<int64_t>(fires.size());
  if (size == 0)
    return schedule;

  // Longest p-periodic suffix: fires[t] == fires[t - p] for every t >= start.
  int64_t bestStart = size;
  int64_t bestPeriod = 0;
  for (int64_t period = 1; period <= kMaxStaticPeriod && period < size;
       ++period) {
    int64_t start = period;
    for (int64_t t = size - 1; t >= period; --t) {
      if (fires[t] != fires[t - period]) {
        start = t + 1;
        break;
      }
    }
    const int64_t length = size - start;
    // Require enough repetitions to be worth the arithmetic tail check.
    if (length < 4 * period || length < 8)
      continue;
    // Smallest period wins: ascending `period` and strictly longer suffixes.
    if (length > size - bestStart) {
      bestStart = start;
      bestPeriod = period;
    }
  }

  if (bestPeriod == 0) {
    appendRuns(fires, 0, size, schedule.prefixRanges);
    return schedule;
  }

  appendRuns(fires, 0, bestStart, schedule.prefixRanges);
  for (int64_t t = bestStart; t < size; ++t) {
    if (!fires[t])
      continue;
    // The suffix is `bestPeriod`-periodic, so its first firing step fixes the
    // phase of the whole tail; steps between bestStart and it never fire.
    schedule.phase = t;
    schedule.period = bestPeriod;
    return schedule;
  }
  // Periodic tail without a firing step: the block is dead after the prefix.
  return schedule;
}

// Simulates the deterministic block schedule of every core.
//
// Read discipline, mirroring the lockstep counter semantics: a value written
// by the peer core is read from the step-start snapshot, a value written by the
// reading core itself is read live. Cross-core group credits are single writer
// by construction (`produced` by the producer block, `consumed` by the consumer
// block), so both cores' progress is well defined without modelling timing.
class Simulator {
public:
  Simulator(const StaticScheduleInput &input) : input(input) {}

  bool run(llvm::SmallVectorImpl<StaticCoreSchedule> &out);

private:
  int slotFor(int group);
  bool gatePasses(const StaticBlockGate &gate, int core);
  void applyExecution(const StaticBlockGate &gate, int core);

  const StaticScheduleInput &input;
  llvm::DenseMap<int, int> slotOf;
  // Per group slot: live values, and the step-start snapshot of peer writes.
  llvm::SmallVector<int64_t> produced;
  llvm::SmallVector<int64_t> consumed;
  llvm::SmallVector<int64_t> producedSnap;
  llvm::SmallVector<int64_t> consumedSnap;
  llvm::SmallVector<int64_t> credit;
  llvm::SmallVector<llvm::SmallVector<int64_t>> vars;
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<char>>> fires;
};

int Simulator::slotFor(int group) {
  auto [it, inserted] = slotOf.try_emplace(group, slotOf.size());
  return it->second;
}

bool Simulator::gatePasses(const StaticBlockGate &gate, int core) {
  for (int group : gate.crossInGroups) {
    int slot = slotFor(group);
    if (producedSnap[slot] - consumed[slot] <= 0)
      return false;
  }
  for (int group : gate.crossOutGroups) {
    int slot = slotFor(group);
    if (produced[slot] - consumedSnap[slot] >= credit[slot])
      return false;
  }
  for (int var : gate.intraInVars) {
    if (vars[core][var] <= 0)
      return false;
  }
  for (auto [var, limit] : gate.intraOutVars) {
    if (vars[core][var] >= limit)
      return false;
  }
  for (int var : gate.tensorInVars) {
    if (vars[core][var] != 1)
      return false;
  }
  for (int var : gate.tensorOutVars) {
    if (vars[core][var] != 0)
      return false;
  }
  if (gate.counterVar >= 0) {
    int64_t counter = vars[core][gate.counterVar];
    if (counter >= input.upperBound)
      return false;
  }
  // flowOpt gates this block on the source node's counter, not on its own.
  if (gate.flowOptVar >= 0 &&
      vars[core][gate.flowOptVar] < gate.flowOptMinCounter)
    return false;
  return true;
}

void Simulator::applyExecution(const StaticBlockGate &gate, int core) {
  for (int group : gate.crossInGroups)
    ++consumed[slotFor(group)];
  for (int group : gate.crossOutGroups)
    ++produced[slotFor(group)];
  for (auto [var, delta] : gate.varUpdates)
    vars[core][var] += delta;
  if (gate.counterVar >= 0)
    vars[core][gate.counterVar] += input.step;
}

bool Simulator::run(llvm::SmallVectorImpl<StaticCoreSchedule> &out) {
  if (input.trip <= 0 || input.step <= 0 || input.cores.empty())
    return false;

  // Allocate group slots first so every gate sees a stable slot id.
  for (const StaticCoreModel &model : input.cores) {
    for (const StaticBlockGate &gate : model.blocks) {
      for (int group : gate.crossInGroups)
        slotFor(group);
      for (int group : gate.crossOutGroups)
        slotFor(group);
    }
  }

  produced.assign(slotOf.size(), 0);
  consumed.assign(slotOf.size(), 0);
  credit.assign(slotOf.size(), 1);
  for (auto [group, buffers] : input.groupCredit) {
    if (!slotOf.count(group))
      continue;
    credit[slotOf[group]] = std::max(1, buffers);
  }

  vars.resize(input.cores.size());
  for (auto [idx, model] : llvm::enumerate(input.cores))
    vars[idx] = model.varInit;

  fires.resize(input.cores.size());
  for (auto [idx, model] : llvm::enumerate(input.cores)) {
    fires[idx].resize(model.blocks.size());
    for (auto &blockFires : fires[idx])
      blockFires.assign(input.trip, 0);
  }

  for (int64_t t = 0; t < input.trip; ++t) {
    producedSnap = produced;
    consumedSnap = consumed;
    for (auto &&[coreIdx, model] : llvm::enumerate(input.cores)) {
      int core = static_cast<int>(coreIdx);
      for (auto &&[blockIdx, gate] : llvm::enumerate(model.blocks)) {
        if (!gatePasses(gate, core))
          continue;
        fires[core][blockIdx][t] = 1;
        applyExecution(gate, core);
      }
    }
  }

  out.resize(input.cores.size());
  for (auto &&[coreIdx, model] : llvm::enumerate(input.cores)) {
    out[coreIdx].blocks.resize(model.blocks.size());
    for (auto &&[blockIdx, gate] : llvm::enumerate(model.blocks))
      out[coreIdx].blocks[blockIdx] = compactSchedule(fires[coreIdx][blockIdx]);
  }
  return true;
}

// Creates an integer constant of `type` (integer or index).
Value createIntConst(OpBuilder &builder, Location loc, Type type,
                     int64_t value) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantOp>(
        loc, type, builder.getIntegerAttr(intType, value));
  if (isa<IndexType>(type))
    return builder.create<arith::ConstantOp>(loc, type,
                                             builder.getIndexAttr(value));
  return Value();
}

} // namespace

bool mlir::triton::buildStaticSchedule(
    const StaticScheduleInput &input,
    llvm::SmallVector<StaticCoreSchedule, 2> &out) {
  Simulator simulator(input);
  return simulator.run(out);
}

Value mlir::triton::emitStaticStepPredicate(
    OpBuilder &builder, Location loc, Value iv, int64_t lowerBound,
    int64_t step, const StaticBlockSchedule &schedule) {
  Type type = iv.getType();
  llvm::SmallVector<Value> clauses;

  for (const IvrRange &range : schedule.prefixRanges) {
    Value begin =
        createIntConst(builder, loc, type, lowerBound + step * range.begin);
    Value end =
        createIntConst(builder, loc, type, lowerBound + step * range.end);
    Value lower = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                iv, begin);
    Value upper =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iv, end);
    clauses.push_back(builder.create<arith::AndIOp>(loc, lower, upper));
  }

  if (schedule.period > 0) {
    // `phase` is the first firing step of the periodic tail: every later firing
    // step is congruent to it modulo `period`, and the steps before it either
    // belong to prefixRanges or never fire. Guarding at `phase` therefore drops
    // no firing step and keeps `iv - base` non-negative, which the unsigned
    // remainder needs.
    Value base =
        createIntConst(builder, loc, type, lowerBound + step * schedule.phase);
    Value stride = createIntConst(builder, loc, type, step * schedule.period);
    Value zero = createIntConst(builder, loc, type, 0);
    Value reached =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iv, base);
    Value delta = builder.create<arith::SubIOp>(loc, iv, base);
    Value remainder = builder.create<arith::RemUIOp>(loc, delta, stride);
    Value aligned = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  remainder, zero);
    clauses.push_back(builder.create<arith::AndIOp>(loc, reached, aligned));
  }

  if (clauses.empty())
    return builder.create<arith::ConstantIntOp>(loc, 0, /*width=*/1);
  Value combined = clauses.front();
  for (Value clause : llvm::ArrayRef<Value>(clauses).drop_front())
    combined = builder.create<arith::OrIOp>(loc, combined, clause);
  return combined;
}
