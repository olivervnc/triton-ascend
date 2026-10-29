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

#ifndef TRITON_ADAPTER_STATIC_SCHEDULE_H
#define TRITON_ADAPTER_STATIC_SCHEDULE_H

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Location;
class OpBuilder;
class Value;
namespace triton {

// Gating and update semantics of one ssbuffer.if, mirrored field by field from
// what UpdateConditionInfoPass emits in the legacy (counter based) form.
//
// A block executes when every gate below allows it; execution then applies
// `varUpdates` and bumps the block counter by `step`.
//
// Variable ids index StaticCoreModel::varInit of the same core. Cross-core
// group ids are module wide ssbuffer.crossCoreDeps group ids.
struct StaticBlockGate {
  // Consumer side of a cross-core buffer: requires produced - consumed > 0.
  llvm::SmallVector<int> crossInGroups;
  // Producer side of a cross-core buffer: requires produced - consumed < N.
  llvm::SmallVector<int> crossOutGroups;
  llvm::SmallVector<int> intraInVars; // var > 0
  // (var, limit) pairs: requires var < limit, limit is the group op count.
  llvm::SmallVector<std::pair<int, int>> intraOutVars;
  llvm::SmallVector<int> tensorInVars;  // var == 1
  llvm::SmallVector<int> tensorOutVars; // var == 0
  // Block counter iter_arg: requires var < upperBound. -1 when absent.
  int counterVar = -1;
  // flowOpt gate: additionally requires the *source* block's counter variable
  // to have reached flowOptMinCounter. `flowOptVar < 0` disables the gate.
  int flowOptVar = -1;
  int64_t flowOptMinCounter = 0;
  // Effective per-variable update on execution (+1 / -1), already reduced with
  // the same precedence the emitter uses (tensorOut > tensorIn > intraOut >
  // intraIn, one entry per variable).
  llvm::SmallVector<std::pair<int, int>> varUpdates;
};

// One core's block sequence for one main loop, in program order.
struct StaticCoreModel {
  // StaticBlockGate is too large for inline storage, so keep it heap only.
  llvm::SmallVector<StaticBlockGate, 0> blocks;
  // Initial value of every control variable. Intra-core deps start at 0,
  // tensor iter_arg flags at 1, block counters at the loop lower bound.
  llvm::SmallVector<int64_t> varInit;
};

// Input of the schedule simulation.
//
// One or two cores participate. Two cores are modelled jointly because
// cross-core group credits are written by one side and read by the other; a
// single core's schedule cannot be derived without the peer's block sequence.
struct StaticScheduleInput {
  int64_t trip = 0;       // stretched iteration count
  int64_t lowerBound = 0; // loop lower bound (block counter seed)
  int64_t upperBound = 0; // pre-stretch upper bound (block counter gate)
  int64_t step = 1;       // loop step
  llvm::SmallVector<StaticCoreModel, 2> cores;
  // Cross-core group id -> number of buffers (credit bound).
  llvm::DenseMap<int, int> groupCredit;
};

// [begin, end) over stretched iteration steps.
struct IvrRange {
  int64_t begin = 0;
  int64_t end = 0;
};

// Firing steps of one block: the union of `prefixRanges` and one arithmetic
// progression {phase, phase + period, ...}. `period == 0` means "no periodic
// part", and a schedule with no prefix range and no period is a block that
// never executes.
struct StaticBlockSchedule {
  llvm::SmallVector<IvrRange> prefixRanges;
  int64_t period = 0;
  int64_t phase = 0;
};

struct StaticCoreSchedule {
  // Indexed like StaticCoreModel::blocks.
  llvm::SmallVector<StaticBlockSchedule, 0> blocks;
};

// Largest period the compaction pass will detect. Rotation schedules are
// periodic with a period bounded by the number of blocks plus the buffer
// depth, so this only needs to cover small values.
constexpr int64_t kMaxStaticPeriod = 16;

// Simulates the deterministic schedule of every modelled core. Returns false
// when the input is not simulable (empty/malformed model); callers must then
// keep the legacy counter based conditions.
bool buildStaticSchedule(const StaticScheduleInput &input,
                         llvm::SmallVector<StaticCoreSchedule, 2> &out);

// Emits `iv` belonging to the firing set as arith ops. `lowerBound` / `step`
// map a schedule step `t` to the induction variable value `lb + step * t`.
Value emitStaticStepPredicate(OpBuilder &builder, Location loc, Value iv,
                              int64_t lowerBound, int64_t step,
                              const StaticBlockSchedule &schedule);

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_STATIC_SCHEDULE_H
