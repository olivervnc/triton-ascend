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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <pthread.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

#include "DynamicCVPipeline/Common/FallbackHelper.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "triton/Tools/Sys/GetEnv.hpp"

static constexpr const char *DEBUG_TYPE = "add-dynamic-cv-pipeline";
static constexpr unsigned MAX_RETRY_TIMES = 2;
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

static std::optional<int64_t> getErrorCode(ModuleOp moduleOp) {
  auto errCodeAttr =
      moduleOp->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
  return errCodeAttr ? std::optional<int64_t>(errCodeAttr.getInt())
                     : std::nullopt;
}

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
                                 DenseMap<int, scope::ScopeOp> &scopeOps) {
  if (auto scopeOp = packScopeOp(ops)) {
    // Safety: holdingBlockId is not null
    scopeOp->setAttr(CVPipeline::kBlockId, blockId);
    scopeOps[blockId.getInt()] = scopeOp;
  }
}

DenseMap<int, scope::ScopeOp> packBlocks(Block &block) {
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
      packScopeOpOfBlockId(holdingGroup, holdingBlockId, scopeOps);
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
  packScopeOpOfBlockId(holdingGroup, holdingBlockId, scopeOps);
  return scopeOps;
}

struct ForWithAddArgs {
  scf::ForOp forOp;
  llvm::SmallVector<BlockArgument> extraBargs;
};

ForWithAddArgs addIterArgToExistingFor(mlir::scf::ForOp oldForOp,
                                       ValueRange newInitValues) {
  mlir::Location loc = oldForOp.getLoc();

  // 1. 组合旧的 Inits 和新增的 Init
  llvm::SmallVector<mlir::Value> newInits(oldForOp.getInitArgs());
  newInits.append(newInitValues.begin(), newInitValues.end());

  IRRewriter rewriter(oldForOp);

  auto numNewVals = newInitValues.size();

  // 2. 创建新的 ForOp
  auto newForOp = rewriter.create<mlir::scf::ForOp>(
      loc, oldForOp.getLowerBound(), oldForOp.getUpperBound(),
      oldForOp.getStep(), newInits,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        mlir::IRMapping mapping;
        // 映射 induction variable
        mapping.map(oldForOp.getInductionVar(), iv);

        // 映射已有的 iter_args (注意 iterArgs 最后一个是新增的)
        for (auto [oldArg, newArg] :
             llvm::zip(oldForOp.getRegionIterArgs(),
                       iterArgs.drop_back(numNewVals))) {
          mapping.map(oldArg, newArg);
        }

        mlir::Value newIterArg = iterArgs.back(); // 这是新增加的 block argument

        // 克隆原循环体内的除 YieldOp 外的所有操作
        auto *oldYield = oldForOp.getBody()->getTerminator();
        for (auto &op : oldForOp.getBody()->without_terminator()) {
          b.clone(op, mapping);
        }

        // 组合旧的 yield operands 和新的 yield operand
        llvm::SmallVector<mlir::Value> newYieldOperands;
        for (mlir::Value oldYieldOperand : oldYield->getOperands()) {
          newYieldOperands.push_back(mapping.lookupOrDefault(oldYieldOperand));
        }
        newYieldOperands.append(newInitValues.begin(), newInitValues.end());

        b.create<mlir::scf::YieldOp>(l, newYieldOperands);
      });

  // 3. 用新 ForOp 对应的结果替换旧 ForOp 的所有使用者
  rewriter.replaceOp(oldForOp, newForOp.getResults().drop_back(numNewVals));

  llvm::SmallVector<BlockArgument> extraBargs(
      newForOp.getRegionIterArgs().take_back(numNewVals));
  return {newForOp, std::move(extraBargs)};
}

class PackScopePass
    : public PassWrapper<PackScopePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackScopePass)

  // Constructor
  PackScopePass() = default;

  // Run the pass
  void runOnOperation() override {
    auto module = getOperation();
    module.walk<WalkOrder::PreOrder>([](scf::ForOp forOp) {
      if (!forOp->hasAttr(CVPipeline::kMainLoop)) {
        return WalkResult::advance();
      }
      auto scopeOps = packBlocks(*forOp.getBody());

      OpBuilder builder(forOp);
      Location loc = forOp.getLoc();
      // 此时 builder 插入点在 forOp 外部（或需要的位置）
      Value lb = forOp.getLowerBound();
      Value ub = forOp.getUpperBound();
      Value step = forOp.getStep();
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
      Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);

      Value newUb = builder.create<arith::SubIOp>(loc, ub, step);

      SmallVector<Value> newInits(forOp.getInits());
      newInits.push_back(zero);                                 // counter
      auto forResults = addIterArgToExistingFor(forOp, {zero}); // add counter
      forOp = forResults.forOp;
      BlockArgument counter = forResults.extraBargs.front();
      forOp.setUpperBound(newUb);

      SmallVector<int> prologueBlockIds{3, 12, 13, 14};
      SmallVector<scope::ScopeOp> prologueBlocks;
      for (auto blockId : prologueBlockIds) {
        if (auto scopeOp = scopeOps.lookup(blockId)) {
          prologueBlocks.push_back(scopeOp);
        }
      }
      DenseSet<scope::ScopeOp> prologueBlockSet(prologueBlocks.begin(),
                                                prologueBlocks.end());

      {
        builder.setInsertionPointToStart(forOp.getBody());
        Value remainder = builder.create<arith::RemSIOp>(loc, counter, two);
        Value isEven = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, remainder, zero);
        builder.create<scf::IfOp>(
            loc, isEven,
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::YieldOp>(loc, zero);
            },
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::YieldOp>(loc, two);
            });
        auto addIOp = builder.create<arith::AddIOp>(
            loc, forOp.getInductionVar(), forOp.getStep());
        addIOp->setAttr("prologued induction", builder.getUnitAttr());
        Value newCounter = builder.create<arith::AddIOp>(loc, counter, one);
        forOp.getTiedLoopYieldedValue(counter)->set(newCounter);
      }

      {
        builder.setInsertionPointAfter(forOp);
        Value remainder = builder.create<arith::RemSIOp>(
            loc, forOp.getTiedLoopResult(counter), two);
        Value isEven = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, remainder, zero);
        builder.create<scf::IfOp>(
            loc, isEven,
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::YieldOp>(loc, zero);
            },
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::YieldOp>(loc, two);
            });
      }

      return WalkResult::skip();
    });
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-pack-scopeop";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scope::ScopeDialect>();
  }
};

} // namespace

static inline void addPasses(OpPassManager &pm) {
  // pm.addPass(createPreCheckAvailablePass());
  // pm.addPass(createStandardizeOpPass());
  // pm.addPass(createPlanComputeBlockPass());
  // pm.addPass(createComputeBlockOptPass());
  pm.addPass(createSplitDataflowPass());
  pm.addPass(std::make_unique<PackScopePass>());
  // pm.addPass(createAnalyzeDataFlowPass());
  // pm.addPass(createAllocMultiCachePass());
  // pm.addPass(createAddControlFlowConditionPass());
  // pm.addPass(createSeparateMemoryFromComputePass());
  // pm.addPass(createRemoveSsbufAttrPass());
}

// must collect all sub-passes since they now are not added to pipeline
void AddDynamicCVPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  Base::getDependentDialects(registry);
  OpPassManager tempPM(ModuleOp::getOperationName());
  addPasses(tempPM);
  tempPM.getDependentDialects(registry);
}

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

static void checkAndDisableVfSub(ModuleOp module) {
  static constexpr llvm::StringLiteral kDisableVfSubKernels[1]{
      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"};
  module->walk([=](func::FuncOp funcOp) {
    if (llvm::is_contained(kDisableVfSubKernels, funcOp.getSymName())) {
      CVPipeline::setFallbackAttr(module,
                                  CVPipeline::ERRCODE_DISABLE_VF_SUBSTITUTION);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

void AddDynamicCVPipelinePass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());
  compileOn91095Flag = this->compileOn91095;

  LDBG("Enter pass");
  moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);

  if (!compileOn91095Flag) {
    llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
    return;
  }

  ScopedDiagnosticHandler handler(&getContext(), [&](Diagnostic &diag) {
    // LLVM_DEBUG({
    // In debug mode, continue to other handlers, i.e. print to stderr
    return llvm::failure();
    // });

    // otherwise prohibit any other handler
    return llvm::success();
  });

  for (unsigned attempt = 0; attempt < MAX_RETRY_TIMES; ++attempt) {
    // restore() consumes the saved region bodies. Each attempt needs its own
    // snapshot, taken before changing buffer counts for the retry.
    CVPipeline::FallbackHelper fallback(moduleOp);
    if (attempt > 0) {
      BufferCountManager bufferCountManager(moduleOp);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::IntraCore,
                                        2);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::InterCore,
                                        1);
    }

    // Do not reuse pass instances or partially transformed IR on retry.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    if (failed(mlir::applyPassManagerCLOptions(pm))) {
      LDBG("Failed to apply cli options - running in python");
    }

    if (tools::getBoolEnv("MLIR_ENABLE_DUMP")) {
      pm.enableIRPrinting();
    }

    addPasses(pm);

    // run passes in separate pm, instead of the pipeline to suppress reproducer
    auto result = pm.run(moduleOp);
    auto errCode = getErrorCode(moduleOp);
    if (succeeded(result) && !errCode.has_value()) {
      checkAndDisableVfSub(moduleOp);
      LDBG("Process successfully");
      return;
    }

    if (errCode == CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED) {
      if (attempt + 1 < MAX_RETRY_TIMES) {
        LDBG("Tuple-buffer failed; Retrying with tuple preload disabled.");
        fallback.restore();
        moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);
        continue;
      }
      // Consume repeated retry requests instead of leaking code 3 to callers.
      errCode = CVPipeline::ERRCODE_FAILED;
    }

    if (!errCode.has_value()) {
      moduleOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                              << "Unexpected pass failure (no fallback attr "
                                 "set); fallback to compilation without "
                                 "dynamic CV pipeline.";
    } else if (errCode == CVPipeline::ERRCODE_IGNORED) {
      // This is an expected fallback: do not attach the full module IR.
      mlir::emitWarning(moduleOp->getLoc())
          << "[" << DEBUG_TYPE << "] "
          << "Kernel not applicable for dynamic CV pipeline "
             "(no matmul / already scope-optimized / unsupported "
             "pattern); falling back to standard compilation.";
    } else {
      moduleOp->emitWarning()
          << "[" << DEBUG_TYPE << "] " << "Pass failed (errcode=" << *errCode
          << "); "
             "falling back to compilation without "
             "dynamic CV pipeline.";
    }

    fallback.restore();
    moduleOp->setAttr(CVPipeline::ERRCODE_ATTR,
                      builder.getI32IntegerAttr(
                          errCode.value_or(CVPipeline::ERRCODE_FAILED)));
    return;
  }

  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options) {
  return std::make_unique<AddDynamicCVPipelinePass>(options);
}
