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

#include "ascend/include/DynamicCVPipeline/StaticCVPipeline.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "static-cv-pipeline";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;

namespace {

scope::ScopeOp packScopeOp(ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return nullptr;
  }
  Operation *insertionPoint = *ops.begin();

  DenseSet<Operation *> allOps;
  for (auto *op : ops) {
    op->walk([&](Operation *subOp) { allOps.insert(subOp); });
  }

  llvm::SetVector<Value> escapedValues;
  for (auto *op : ops) {
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (allOps.contains(user)) {
          continue;
        }
        escapedValues.insert(result);
        break;
      }
    }
  }

  OpBuilder builder(insertionPoint);
  builder.setInsertionPoint(insertionPoint);
  Location loc = insertionPoint->getLoc();
  ValueRange redirectValues(escapedValues.getArrayRef());
  TypeRange types = redirectValues.getTypes();
  auto scopeOp = builder.create<scope::ScopeOp>(loc, types);
  for (auto [origVal, scopeRes] :
       llvm::zip(redirectValues, scopeOp->getResults())) {
    origVal.replaceUsesWithIf(scopeRes, [&](OpOperand &operand) {
      return !allOps.contains(operand.getOwner());
    });
  }

  auto *block = &scopeOp.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToEnd(block);
  auto returnOp =
      builder.create<scope::ReturnOp>(loc, escapedValues.getArrayRef());
  for (auto *op : ops) {
    op->moveBefore(returnOp);
  }
  return scopeOp;
}

inline void packScopeOpOfBlockId(ArrayRef<Operation *> ops, IntegerAttr blockId,
                                 DenseMap<int, scope::ScopeOp> &scopeOps,
                                 SmallVector<int> &blockOrder) {
  if (auto scopeOp = packScopeOp(ops)) {
    scopeOp->setAttr(CVPipeline::kBlockId, blockId);
    int bid = blockId.getInt();
    scopeOps[bid] = scopeOp;
    blockOrder.push_back(bid);
  }
}

DenseMap<int, scope::ScopeOp> packBlocks(Block &block,
                                         SmallVector<int> &blockOrder) {
  SmallVector<Operation *> holdingGroup;
  DenseMap<int, scope::ScopeOp> scopeOps;
  IntegerAttr holdingBlockId;
  for (auto &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    auto currBlockId = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!holdingBlockId && currBlockId) {
      holdingBlockId = currBlockId;
    }
    if (!holdingBlockId) {
      continue;
    }
    if (holdingBlockId != currBlockId) {
      packScopeOpOfBlockId(holdingGroup, holdingBlockId, scopeOps, blockOrder);
      holdingBlockId = currBlockId;
      holdingGroup.clear();
    }
    for (auto operand : op.getOperands()) {
      auto allocOp =
          llvm::dyn_cast_if_present<memref::AllocOp>(operand.getDefiningOp());
      if (!allocOp) {
        continue;
      }
      if (operand.getNumUses() != 2) {
        continue;
      }
      auto markOpOpt = utils::getAnnotateOpWithAttr(
          operand, CVPipeline::kTightlyCoupledBufferAttr);
      if (!markOpOpt) {
        continue;
      }
      auto markOp = markOpOpt.value();
      markOp->moveBefore(&op);
      allocOp->moveBefore(markOp);
      holdingGroup.push_back(allocOp);
      holdingGroup.push_back(markOp);
    }
    holdingGroup.push_back(&op);
  }
  packScopeOpOfBlockId(holdingGroup, holdingBlockId, scopeOps, blockOrder);
  return scopeOps;
}

struct ForWithAddArgs {
  scf::ForOp forOp;
  llvm::SmallVector<BlockArgument> extraBargs;
};

ForWithAddArgs addIterArgsToFor(mlir::scf::ForOp oldForOp,
                                ValueRange newInitValues) {
  mlir::Location loc = oldForOp.getLoc();
  llvm::SmallVector<mlir::Value> newInits(oldForOp.getInitArgs());
  newInits.append(newInitValues.begin(), newInitValues.end());

  IRRewriter rewriter(oldForOp);
  auto numNewVals = newInitValues.size();

  auto newForOp = rewriter.create<mlir::scf::ForOp>(
      loc, oldForOp.getLowerBound(), oldForOp.getUpperBound(),
      oldForOp.getStep(), newInits,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        mlir::IRMapping mapping;
        mapping.map(oldForOp.getInductionVar(), iv);

        for (auto [oldArg, newArg] :
             llvm::zip(oldForOp.getRegionIterArgs(),
                       iterArgs.drop_back(numNewVals))) {
          mapping.map(oldArg, newArg);
        }

        auto *oldYield = oldForOp.getBody()->getTerminator();
        for (auto &op : oldForOp.getBody()->without_terminator()) {
          b.clone(op, mapping);
        }

        llvm::SmallVector<mlir::Value> newYieldOperands;
        for (mlir::Value oldYieldOperand : oldYield->getOperands()) {
          newYieldOperands.push_back(mapping.lookupOrDefault(oldYieldOperand));
        }
        newYieldOperands.append(newInitValues.begin(), newInitValues.end());

        b.create<mlir::scf::YieldOp>(l, newYieldOperands);
      });

  rewriter.replaceOp(oldForOp, newForOp.getResults().drop_back(numNewVals));

  llvm::SmallVector<BlockArgument> extraBargs(
      newForOp.getRegionIterArgs().take_back(numNewVals));
  return {newForOp, std::move(extraBargs)};
}

/// Helper to clone a scopeOp with flag delta and tightly coupled buffer
/// replacements.
/// flagDelta = 0: even iteration (+0 flag, original tcb buffer)
/// flagDelta = 5: odd iteration (+5 flag, +10 tcb buffer)
scope::ScopeOp cloneScopeOpWithReplacements(
    OpBuilder &builder, Location loc, scope::ScopeOp origScope,
    const IRMapping &baseMapping, int flagDelta,
    const DenseMap<Value, Value> &oddBufferMap) {
  IRMapping mapping(baseMapping);

  auto newScope = builder.create<scope::ScopeOp>(
      loc, origScope.getResultTypes());
  if (auto blockId = origScope->getAttr(CVPipeline::kBlockId)) {
    newScope->setAttr(CVPipeline::kBlockId, blockId);
  }

  OpBuilder::InsertionGuard guard(builder);
  auto *newBlock = &newScope.getBodyRegion().emplaceBlock();
  builder.setInsertionPointToEnd(newBlock);

  for (auto &innerOp : origScope.getBodyRegion().front()) {
    if (auto returnOp = dyn_cast<scope::ReturnOp>(innerOp)) {
      SmallVector<Value> returnOperands;
      for (auto operand : returnOp.getOperands()) {
        returnOperands.push_back(mapping.lookupOrDefault(operand));
      }
      builder.create<scope::ReturnOp>(loc, returnOperands);
      continue;
    }

    if (flagDelta != 0) {
      if (auto setOp = dyn_cast<hivm::SyncBlockSetOp>(innerOp)) {
        if (auto staticFlag = setOp.getStaticFlagId()) {
          int64_t oldFlag = staticFlag->getInt();
          auto newSet = builder.create<hivm::SyncBlockSetOp>(
              setOp.getLoc(), setOp.getTcoreType(), setOp.getTpipe(),
              setOp.getPipe(), builder.getI64IntegerAttr(oldFlag + flagDelta));
          if (auto blockId = setOp->getAttr(CVPipeline::kBlockId))
            newSet->setAttr(CVPipeline::kBlockId, blockId);
          if (auto tid = setOp->getAttr(CVPipeline::kTransferId))
            newSet->setAttr(CVPipeline::kTransferId, tid);
          continue;
        }
      } else if (auto waitOp = dyn_cast<hivm::SyncBlockWaitOp>(innerOp)) {
        if (auto staticFlag = waitOp.getStaticFlagId()) {
          int64_t oldFlag = staticFlag->getInt();
          auto newWait = builder.create<hivm::SyncBlockWaitOp>(
              waitOp.getLoc(), waitOp.getTcoreType(), waitOp.getTpipe(),
              waitOp.getPipe(), builder.getI64IntegerAttr(oldFlag + flagDelta));
          if (auto blockId = waitOp->getAttr(CVPipeline::kBlockId))
            newWait->setAttr(CVPipeline::kBlockId, blockId);
          if (auto tid = waitOp->getAttr(CVPipeline::kTransferId))
            newWait->setAttr(CVPipeline::kTransferId, tid);
          continue;
        }
      } else if (auto markOp = dyn_cast<annotation::MarkOp>(innerOp)) {
        if (auto tcbAttr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
                "hivm.tightly_coupled_buffer")) {
          if (tcbAttr.getId().has_value()) {
            int32_t oldId = tcbAttr.getId().value();
            Value mappedSrc = mapping.lookupOrDefault(markOp.getSrc());
            if (oddBufferMap.count(markOp.getSrc())) {
              mappedSrc = oddBufferMap.lookup(markOp.getSrc());
            }
            auto newMark = builder.create<annotation::MarkOp>(
                markOp.getLoc(), mappedSrc);
            for (auto namedAttr : markOp->getAttrs()) {
              if (namedAttr.getName() == "hivm.tightly_coupled_buffer") {
                newMark->setAttr(
                    "hivm.tightly_coupled_buffer",
                    hivm::HIVMTightlyCoupledBufferAttr::get(
                        builder.getContext(), oldId + 10));
              } else {
                newMark->setAttr(namedAttr.getName(), namedAttr.getValue());
              }
            }
            continue;
          }
        }
      }
    }

    // Remap operands with oddBufferMap if flagDelta != 0
    if (flagDelta != 0) {
      for (auto operand : innerOp.getOperands()) {
        if (oddBufferMap.count(operand) && !mapping.contains(operand)) {
          mapping.map(operand, oddBufferMap.lookup(operand));
        }
      }
    }

    builder.clone(innerOp, mapping);
  }

  return newScope;
}

class StaticCVPipelinePass
    : public PassWrapper<StaticCVPipelinePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticCVPipelinePass)

  StaticCVPipelinePass() = default;

  void runOnOperation() override {
    auto module = getOperation();

    // Check prerequisites & process main loops
    WalkResult result = module.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      if (!forOp->hasAttr(CVPipeline::kMainLoop)) {
        return WalkResult::advance();
      }

      // Check requirement 5: verify no sharing of iter_args across multiple blocks
      for (auto iterArg : forOp.getRegionIterArgs()) {
        DenseSet<int> blockIds;
        for (auto *user : iterArg.getUsers()) {
          Operation *topOp = user;
          while (topOp->getParentOp() && topOp->getParentOp() != forOp) {
            topOp = topOp->getParentOp();
          }
          if (auto bAttr = topOp->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
            blockIds.insert(bAttr.getInt());
          }
        }
        if (blockIds.size() > 1) {
          forOp->emitError(
              "Sharing iter_args across multiple blocks is not supported in static pipeline");
          CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
          return WalkResult::interrupt();
        }
      }

      SmallVector<int> blockOrder;
      auto scopeOps = packBlocks(*forOp.getBody(), blockOrder);
      if (scopeOps.empty()) {
        return WalkResult::advance();
      }

      // Requirement 3: Hardcoded prologue block ids {3, 12, 13, 14}, the rest are epilogue
      const SmallVector<int> hardcodedPrologueIds{3, 12, 13, 14};
      DenseSet<int> prologueIdSet(hardcodedPrologueIds.begin(),
                                  hardcodedPrologueIds.end());

      SmallVector<scope::ScopeOp> prologueScopes;
      SmallVector<scope::ScopeOp> epilogueScopes;

      for (int bid : blockOrder) {
        if (!scopeOps.count(bid))
          continue;
        auto scopeOp = scopeOps[bid];
        if (prologueIdSet.contains(bid)) {
          prologueScopes.push_back(scopeOp);
        } else {
          epilogueScopes.push_back(scopeOp);
        }
      }

      if (prologueScopes.empty() || epilogueScopes.empty()) {
        LDBG("Missing prologue or epilogue blocks, skipping static pipeline");
        return WalkResult::advance();
      }

      OpBuilder builder(forOp);
      Location loc = forOp.getLoc();

      // Collect TCB AllocOps and allocate odd buffers (+10 tcb id)
      DenseMap<Value, Value> tcbOddBufferMap;
      forOp->walk([&](memref::AllocOp allocOp) {
        Value allocRes = allocOp.getResult();
        auto markOpOpt = utils::getAnnotateOpWithAttr(
            allocRes, CVPipeline::kTightlyCoupledBufferAttr);
        if (!markOpOpt)
          return;
        auto markOp = *markOpOpt;
        auto tcbAttr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            "hivm.tightly_coupled_buffer");
        if (!tcbAttr || !tcbAttr.getId().has_value())
          return;

        int32_t tcbId = tcbAttr.getId().value();
        OpBuilder allocBuilder(forOp);
        auto oddAlloc = allocBuilder.create<memref::AllocOp>(
            allocOp.getLoc(), allocOp.getType());
        if (auto blockId = allocOp->getAttr(CVPipeline::kBlockId))
          oddAlloc->setAttr(CVPipeline::kBlockId, blockId);
        if (auto tid = allocOp->getAttr(CVPipeline::kTransferId))
          oddAlloc->setAttr(CVPipeline::kTransferId, tid);

        auto oddMark = allocBuilder.create<annotation::MarkOp>(
            markOp->getLoc(), oddAlloc.getResult());
        for (auto namedAttr : markOp->getAttrs()) {
          if (namedAttr.getName() == "hivm.tightly_coupled_buffer") {
            oddMark->setAttr("hivm.tightly_coupled_buffer",
                             hivm::HIVMTightlyCoupledBufferAttr::get(
                                 allocBuilder.getContext(), tcbId + 10));
          } else {
            oddMark->setAttr(namedAttr.getName(), namedAttr.getValue());
          }
        }
        tcbOddBufferMap[allocRes] = oddAlloc.getResult();
      });

      // Analyze intra-core data dependencies from Prologue to Epilogue
      DenseSet<Operation *> epiScopeSet;
      for (auto s : epilogueScopes)
        epiScopeSet.insert(s.getOperation());

      SmallVector<Value> proToEpiValues;
      for (auto proScope : prologueScopes) {
        for (auto res : proScope.getResults()) {
          for (auto *user : res.getUsers()) {
            Operation *topUser = user;
            while (topUser->getParentOp() && topUser->getParentOp() != forOp) {
              topUser = topUser->getParentOp();
            }
            if (epiScopeSet.contains(topUser)) {
              proToEpiValues.push_back(res);
              break;
            }
          }
        }
      }

      // Loop constants
      Value lb = forOp.getLowerBound();
      Value ub = forOp.getUpperBound();
      Value step = forOp.getStep();
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
      Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
      Value newUb = builder.create<arith::SubIOp>(loc, ub, step);

      // --- Pre-loop: Prologue (iter 0, counter = 0, even) ---
      IRMapping preMapping;
      preMapping.map(forOp.getInductionVar(), lb);
      for (auto [oldArg, init] :
           llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
        preMapping.map(oldArg, init);
      }

      SmallVector<Value> prePrologueOutputs;
      for (auto proScope : prologueScopes) {
        auto clonedScope = cloneScopeOpWithReplacements(
            builder, loc, proScope, preMapping, /*flagDelta=*/0, tcbOddBufferMap);
        for (auto [origRes, clonedRes] :
             llvm::zip(proScope.getResults(), clonedScope.getResults())) {
          preMapping.map(origRes, clonedRes);
          if (llvm::is_contained(proToEpiValues, origRes)) {
            prePrologueOutputs.push_back(clonedRes);
          }
        }
      }

      // --- Add iter args to forOp: [counter, proToEpiValues...] ---
      SmallVector<Value> extraInits;
      extraInits.push_back(zero); // counter init
      extraInits.append(prePrologueOutputs.begin(), prePrologueOutputs.end());

      auto forResults = addIterArgsToFor(forOp, extraInits);
      forOp = forResults.forOp;
      forOp.setUpperBound(newUb);

      BlockArgument counterArg = forResults.extraBargs.front();
      ArrayRef<BlockArgument> proToEpiIterArgs =
          ArrayRef<BlockArgument>(forResults.extraBargs).drop_front(1);

      // --- Inside Main Loop ---
      builder.setInsertionPointToStart(forOp.getBody());
      Value remainder = builder.create<arith::RemSIOp>(loc, counterArg, two);
      Value isEven = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, remainder, zero);

      Value prologueIv = builder.create<arith::AddIOp>(
          loc, forOp.getInductionVar(), step);
      prologueIv.getDefiningOp()->setAttr("prologued induction",
                                          builder.getUnitAttr());

      // 1. Epilogue blocks in Main Loop (processing iter i)
      // If isEven is true: flagDelta = 0 (+0 flag, +0 tcb)
      // If isEven is false: flagDelta = 5 (+5 flag, +10 tcb)
      IRMapping loopEpiMapping;
      for (auto [origVal, iterArg] :
           llvm::zip(proToEpiValues, proToEpiIterArgs)) {
        loopEpiMapping.map(origVal, iterArg);
      }
      for (auto [origArg, currentArg] :
           llvm::zip(forOp.getRegionIterArgs().drop_back(extraInits.size()),
                     forOp.getRegionIterArgs().drop_back(extraInits.size()))) {
        loopEpiMapping.map(origArg, currentArg);
      }
      loopEpiMapping.map(forOp.getInductionVar(), forOp.getInductionVar());

      auto ifEpiOp = builder.create<scf::IfOp>(
          loc, TypeRange{}, isEven, /*withElseRegion=*/true);
      {
        OpBuilder thenB = ifEpiOp.getThenBodyBuilder();
        IRMapping thenMapping(loopEpiMapping);
        for (auto epiScope : epilogueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              thenB, loc, epiScope, thenMapping, /*flagDelta=*/0,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(epiScope.getResults(), cloned.getResults())) {
            thenMapping.map(origRes, clonedRes);
          }
        }
      }
      {
        OpBuilder elseB = ifEpiOp.getElseBodyBuilder();
        IRMapping elseMapping(loopEpiMapping);
        for (auto epiScope : epilogueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              elseB, loc, epiScope, elseMapping, /*flagDelta=*/5,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(epiScope.getResults(), cloned.getResults())) {
            elseMapping.map(origRes, clonedRes);
          }
        }
      }

      // 2. Prologue blocks in Main Loop (processing iter i + 1)
      // If isEven is true (next iter is odd): flagDelta = 5 (+5 flag, +10 tcb)
      // If isEven is false (next iter is even): flagDelta = 0 (+0 flag, +0 tcb)
      builder.setInsertionPointAfter(ifEpiOp);
      IRMapping loopProMapping;
      loopProMapping.map(forOp.getInductionVar(), prologueIv);

      SmallVector<Type> proYieldTypes;
      for (auto v : proToEpiValues) {
        proYieldTypes.push_back(v.getType());
      }

      auto ifProOp = builder.create<scf::IfOp>(
          loc, proYieldTypes, isEven, /*withElseRegion=*/true);
      {
        OpBuilder thenB = ifProOp.getThenBodyBuilder();
        IRMapping thenMapping(loopProMapping);
        SmallVector<Value> thenYieldValues;
        for (auto proScope : prologueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              thenB, loc, proScope, thenMapping, /*flagDelta=*/5,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(proScope.getResults(), cloned.getResults())) {
            thenMapping.map(origRes, clonedRes);
            if (llvm::is_contained(proToEpiValues, origRes)) {
              thenYieldValues.push_back(clonedRes);
            }
          }
        }
        thenB.create<scf::YieldOp>(loc, thenYieldValues);
      }
      {
        OpBuilder elseB = ifProOp.getElseBodyBuilder();
        IRMapping elseMapping(loopProMapping);
        SmallVector<Value> elseYieldValues;
        for (auto proScope : prologueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              elseB, loc, proScope, elseMapping, /*flagDelta=*/0,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(proScope.getResults(), cloned.getResults())) {
            elseMapping.map(origRes, clonedRes);
            if (llvm::is_contained(proToEpiValues, origRes)) {
              elseYieldValues.push_back(clonedRes);
            }
          }
        }
        elseB.create<scf::YieldOp>(loc, elseYieldValues);
      }

      // Update counter and loop yields
      builder.setInsertionPointAfter(ifProOp);
      Value newCounter = builder.create<arith::AddIOp>(loc, counterArg, one);
      auto *oldYield = forOp.getBody()->getTerminator();
      oldYield->setOperand(oldYield->getNumOperands() - extraInits.size(),
                           newCounter);
      for (auto [i, proRes] : llvm::enumerate(ifProOp.getResults())) {
        oldYield->setOperand(
            oldYield->getNumOperands() - extraInits.size() + 1 + i, proRes);
      }

      // Erase original scope ops inside loop
      for (auto p : scopeOps) {
        p.second->erase();
      }

      // --- Post-loop: Epilogue (last iteration) ---
      builder.setInsertionPointAfter(forOp);
      Value finalCounter =
          forOp.getTiedLoopResult(counterArg);
      Value postRemainder =
          builder.create<arith::RemSIOp>(loc, finalCounter, two);
      Value postIsEven = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, postRemainder, zero);

      IRMapping postEpiMapping;
      for (auto [i, origVal] : llvm::enumerate(proToEpiValues)) {
        Value loopOut = forOp.getResults()[forOp.getNumResults() -
                                           extraInits.size() + 1 + i];
        postEpiMapping.map(origVal, loopOut);
      }
      for (auto [origArg, loopRes] :
           llvm::zip(forOp.getRegionIterArgs().drop_back(extraInits.size()),
                     forOp.getResults().drop_back(extraInits.size()))) {
        postEpiMapping.map(origArg, loopRes);
      }

      auto ifPostEpiOp = builder.create<scf::IfOp>(
          loc, TypeRange{}, postIsEven, /*withElseRegion=*/true);
      {
        OpBuilder thenB = ifPostEpiOp.getThenBodyBuilder();
        IRMapping thenMapping(postEpiMapping);
        for (auto epiScope : epilogueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              thenB, loc, epiScope, thenMapping, /*flagDelta=*/0,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(epiScope.getResults(), cloned.getResults())) {
            thenMapping.map(origRes, clonedRes);
          }
        }
      }
      {
        OpBuilder elseB = ifPostEpiOp.getElseBodyBuilder();
        IRMapping elseMapping(postEpiMapping);
        for (auto epiScope : epilogueScopes) {
          auto cloned = cloneScopeOpWithReplacements(
              elseB, loc, epiScope, elseMapping, /*flagDelta=*/5,
              tcbOddBufferMap);
          for (auto [origRes, clonedRes] :
               llvm::zip(epiScope.getResults(), cloned.getResults())) {
            elseMapping.map(origRes, clonedRes);
          }
        }
      }

      return WalkResult::skip();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "static-cv-pipeline";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scope::ScopeDialect, hivm::HIVMDialect,
                    annotation::AnnotationDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createStaticCVPipelinePass() {
  return std::make_unique<StaticCVPipelinePass>();
}

} // namespace triton
} // namespace mlir
