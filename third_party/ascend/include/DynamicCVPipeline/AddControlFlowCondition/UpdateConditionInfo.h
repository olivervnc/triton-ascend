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

#ifndef TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
#define TRITON_ADAPTER_UPDATE_CONDITION_INFO_H

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"

namespace mlir {
namespace triton {

enum class VarUpdateType { INC, DEC };

struct OutputGroupInfo {
  SmallVector<Operation *> outputs;
  SmallVector<Value> inputVars;
};

// Information extracted for each IfOp (either Cube or Vector)
struct IfOpInfo {
  scf::IfOp ifOp;
  int blockId = -1;
  bool isAIC = false; // true if Cube, false if Vector
  Value counter;      // loop counter assigned to this ifOp
  // Cross-core buffer group indices consumed / produced
  SmallVector<int> crossCoreInputValues;
  SmallVector<int> crossCoreOutputValues;
  // Intra-core buffer group indices consumed / produced
  SmallVector<int> intraCoreInputValues;
  SmallVector<int> intraCoreOutputValues;
};

class UpdateConditionInfoPass
    : public PassWrapper<UpdateConditionInfoPass, OperationPass<ModuleOp>> {
public:
  UpdateConditionInfoPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

private:
  // Step 1: Collect main loops and scopes
  LogicalResult findCubeAndVectorLoops(
      ModuleOp module,
      Operation *&cubeScope, scf::ForOp &cubeForOp,
      Operation *&vectorScope, scf::ForOp &vectorForOp);

  // Step 2: Buffer dependency collection
  void collectDependencyBuffers(
      ModuleOp module, SmallVector<Operation *> &mainLoopOps,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &crossCoreBuffers,
      DenseMap<Operation *,
               DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
          &intraCoreBuffersMap);

  // Step 3: Parse IfOp metadata (block_id, dependencies)
  int getInputOutputValues(
      scf::IfOp ifOp,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          crossCoreBuffers,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          intraCoreBuffers,
      SmallVector<int> &crossCoreInputValues,
      SmallVector<int> &crossCoreOutputValues,
      SmallVector<int> &intraCoreInputValues,
      SmallVector<int> &intraCoreOutputValues);

  int buildIdxToVarMap(
      Operation *loopOp,
      const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      DenseMap<int, Value> &idxToVar);

  int buildTensorIterArgIfOpVarMap(Operation *loopOp);

  int buildOutputGroups(
      SmallVector<int> &intraCoreOutputValues,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      DenseMap<int, Value> &idxToVar,
      SmallVector<OutputGroupInfo> &outputGroups);

  // Step 4: Condition collection
  void collectIntraCoreInputConditions(
      OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
      DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
      DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  int collectIntraCoreOutputConditions(
      OpBuilder &builder, Location loc,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  void collectTensorIterArgInputConditions(
      OpBuilder &builder, Location loc, scf::IfOp ifOp,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  void collectTensorIterArgOutputConditions(
      OpBuilder &builder, Location loc, scf::IfOp ifOp,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  int setFlowOptCondition(scf::IfOp currentIfOp, Operation *loopOp,
                          Value counter, Value &flowOptCond);

  // Cross-core conditions using scalar iter_arg tokens (no SSBuffer!)
  void collectCrossCoreTokenConditions(
      OpBuilder &builder, Location loc,
      const SmallVector<int> &crossCoreInputValues,
      const SmallVector<int> &crossCoreOutputValues,
      const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &crossCoreBuffers,
      const DenseMap<int, Value> &crossCoreTokenMap,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  // Step 5: IfOp construction (Real and Dummy)
  scf::IfOp createRealIfOp(
      scf::IfOp oldIfOp, Value combinedCond,
      const SmallVector<Value> &usedVars,
      const DenseMap<Value, VarUpdateType> &varUpdateTypes,
      bool hasCounter, Value counter, Value step);

  scf::IfOp createDummyIfOp(
      OpBuilder &builder, Location loc, int blockId, Value combinedCond,
      const SmallVector<Value> &usedVars,
      const DenseMap<Value, VarUpdateType> &varUpdateTypes,
      bool hasCounter, Value counter, Value step);

  void updateDAGAfterIfOpReplacement(scf::IfOp oldIfOp, scf::IfOp newIfOp);

  DenseMap<Value, Value> controlVarToLatestValue;
  ControlFlowConditionInfo *info = nullptr;
  llvm::DenseMap<scf::IfOp, TensorIterArgIfOpVars> tensorIterArgIfOpVars;
};

std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
