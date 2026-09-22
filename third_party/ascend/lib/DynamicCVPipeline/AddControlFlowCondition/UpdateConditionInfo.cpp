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

#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr int UPDATE_CONDITION_INFO_SUCCESS = 0;
static constexpr int UPDATE_CONDITION_INFO_FAILED = -1;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

namespace mlir {
namespace triton {

using namespace hivm;
using namespace CVPipeline;

static Value getLatestValue(const DenseMap<Value, Value> &map, Value key) {
  auto it = map.find(key);
  return (it != map.end()) ? it->second : key;
}

static Value createIntConst(OpBuilder &builder, Location loc, int64_t val,
                            Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  unsigned width = intType ? intType.getWidth() : 32;
  return builder.create<arith::ConstantIntOp>(loc, val, width).getResult();
}

// Read block id from ssbuffer.if on ifOp. Missing attr is unexpected.
static int getIfBlockId(scf::IfOp ifOp, int &outBlockId) {
  auto attr = ifOp->getAttrOfType<IntegerAttr>(kIf);
  if (!attr) {
    LDBG("ssbuffer.if missing block id on ifOp: " << ifOp);
    return UPDATE_CONDITION_INFO_FAILED;
  }
  outBlockId = static_cast<int>(attr.getInt());
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect ssbuffer.if operations from forOp body.
static void collectSSBufferIfOps(scf::ForOp forOp,
                                 SmallVector<scf::IfOp> &ifOps) {
  forOp.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr(kIf)) {
      ifOps.push_back(ifOp);
    }
  });
}

// Find Cube and Vector scopes and their respective main loops.
LogicalResult UpdateConditionInfoPass::findCubeAndVectorLoops(
    ModuleOp module, Operation *&cubeScope, scf::ForOp &cubeForOp,
    Operation *&vectorScope, scf::ForOp &vectorForOp) {
  cubeScope = nullptr;
  vectorScope = nullptr;
  cubeForOp = nullptr;
  vectorForOp = nullptr;

  auto aiCAttr =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto aivAttr =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);

  module.walk([&](scope::ScopeOp scopeOp) {
    if (!scopeOp->hasAttr("hivm.tcore_type"))
      return WalkResult::advance();
    auto attr = scopeOp->getAttr("hivm.tcore_type");
    if (attr == aiCAttr) {
      cubeScope = scopeOp;
      scopeOp.walk([&](scf::ForOp forOp) {
        if (isMainLoopOp(forOp)) {
          cubeForOp = forOp;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    } else if (attr == aivAttr) {
      vectorScope = scopeOp;
      scopeOp.walk([&](scf::ForOp forOp) {
        if (isMainLoopOp(forOp)) {
          vectorForOp = forOp;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    return WalkResult::advance();
  });

  if (!cubeForOp && !vectorForOp) {
    // Fallback: search any main loops in the module
    module.walk([&](scf::ForOp forOp) {
      if (isMainLoopOp(forOp)) {
        if (!vectorForOp)
          vectorForOp = forOp;
        else if (!cubeForOp)
          cubeForOp = forOp;
      }
    });
  }

  return success();
}

// Collect dependency buffers
void UpdateConditionInfoPass::collectDependencyBuffers(
    ModuleOp module, SmallVector<Operation *> &mainLoopOps,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &crossCoreBuffers,
    DenseMap<Operation *,
             DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
        &intraCoreBuffersMap) {
  int crossCoreIdx = 0;
  module.walk([&](Operation *op) {
    auto it = info->crossCoreDependentMap.find(op);
    if (it != info->crossCoreDependentMap.end()) {
      for (SmallVector<Operation *> &producers : it->second) {
        crossCoreBuffers[crossCoreIdx][op] = producers;
        crossCoreIdx++;
      }
    }
    return WalkResult::advance();
  });

  for (Operation *loopOp : mainLoopOps) {
    if (info->intraCoreDependentMap.count(loopOp)) {
      auto &loopDeps = info->intraCoreDependentMap[loopOp];
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          intraCoreBuffers;
      int intraCoreIdx = 0;
      for (auto &entry : loopDeps) {
        intraCoreBuffers[intraCoreIdx][entry.first] = entry.second;
        intraCoreIdx++;
      }
      intraCoreBuffersMap[loopOp] = intraCoreBuffers;
    }
  }
}

// Helper to build reverse mappings for buffer dependencies
static int buildBufferDependencyMappings(
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &buffers,
    DenseMap<Operation *, SmallVector<int>> &consumerToGroups,
    DenseMap<Operation *, SmallVector<int>> &outputToGroups) {
  for (auto &[groupIdx, deps] : buffers) {
    for (auto &[consumer, producers] : deps) {
      consumerToGroups[consumer].push_back(groupIdx);
      for (Operation *producer : producers) {
        outputToGroups[producer].push_back(groupIdx);
      }
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Analyze input/output buffer groups used in a single ifOp
int UpdateConditionInfoPass::getInputOutputValues(
    scf::IfOp ifOp,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        crossCoreBuffers,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers,
    SmallVector<int> &crossCoreInputValues,
    SmallVector<int> &crossCoreOutputValues,
    SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues) {
  DenseSet<int> crossCoreInputSet;
  DenseSet<int> crossCoreOutputSet;
  DenseSet<int> intraCoreInputSet;
  DenseSet<int> intraCoreOutputSet;

  DenseMap<Operation *, SmallVector<int>> crossCoreOutputToGroups;
  DenseMap<Operation *, SmallVector<int>> intraCoreOutputToGroups;
  DenseMap<Operation *, SmallVector<int>> crossCoreConsumerToGroups;
  DenseMap<Operation *, SmallVector<int>> intraCoreConsumerToGroups;

  if (buildBufferDependencyMappings(crossCoreBuffers, crossCoreConsumerToGroups,
                                    crossCoreOutputToGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (buildBufferDependencyMappings(intraCoreBuffers, intraCoreConsumerToGroups,
                                    intraCoreOutputToGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  ifOp.walk([&](Operation *op) {
    if (op == ifOp)
      return WalkResult::advance();

    if (crossCoreConsumerToGroups.count(op)) {
      for (int idx : crossCoreConsumerToGroups[op])
        crossCoreInputSet.insert(idx);
    }
    if (intraCoreConsumerToGroups.count(op)) {
      for (int idx : intraCoreConsumerToGroups[op])
        intraCoreInputSet.insert(idx);
    }
    if (crossCoreOutputToGroups.count(op)) {
      for (int idx : crossCoreOutputToGroups[op])
        crossCoreOutputSet.insert(idx);
    }
    if (intraCoreOutputToGroups.count(op)) {
      for (int idx : intraCoreOutputToGroups[op])
        intraCoreOutputSet.insert(idx);
    }
    return WalkResult::advance();
  });

  crossCoreInputValues.assign(crossCoreInputSet.begin(),
                              crossCoreInputSet.end());
  crossCoreOutputValues.assign(crossCoreOutputSet.begin(),
                               crossCoreOutputSet.end());
  intraCoreInputValues.assign(intraCoreInputSet.begin(),
                              intraCoreInputSet.end());
  intraCoreOutputValues.assign(intraCoreOutputSet.begin(),
                               intraCoreOutputSet.end());

  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::buildIdxToVarMap(
    Operation *loopOp,
    const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar) {
  idxToVar.clear();
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp)
    return UPDATE_CONDITION_INFO_FAILED;
  auto regionIterArgs = forOp.getRegionIterArgs();
  int iterArgNum = static_cast<int>(regionIterArgs.size());

  if (!info->innerDepConds.count(loopOp))
    return UPDATE_CONDITION_INFO_SUCCESS;

  const auto &innerDepIndices = info->innerDepConds[loopOp];
  int varIdx = 0;
  for (const auto &entry : intraCoreBuffers) {
    int idx = entry.first;
    if (varIdx >= static_cast<int>(innerDepIndices.size()))
      break;
    int argIdx = innerDepIndices[varIdx];
    if (argIdx >= 0 && argIdx < iterArgNum) {
      idxToVar[idx] = regionIterArgs[argIdx];
    }
    varIdx++;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::buildTensorIterArgIfOpVarMap(Operation *loopOp) {
  tensorIterArgIfOpVars.clear();
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp)
    return UPDATE_CONDITION_INFO_SUCCESS;
  if (!info->tensorIterArgDepsMap.count(loopOp) ||
      !info->tensorIterArgIndicesMap.count(loopOp)) {
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  auto &depsVec = info->tensorIterArgDepsMap[loopOp];
  auto &indicesMap = info->tensorIterArgIndicesMap[loopOp];
  auto regionIterArgs = forOp.getRegionIterArgs();

  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> producerVars;
  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> consumerVars;

  for (auto &depEntry : depsVec) {
    Value origIterArg = depEntry.iterArg;
    if (!indicesMap.count(origIterArg))
      continue;
    SmallVector<int> &argIndices = indicesMap[origIterArg];
    if (depEntry.consumers.size() != argIndices.size())
      continue;

    llvm::DenseMap<scf::IfOp, Value> consumerToVar;
    for (size_t i = 0; i < depEntry.consumers.size(); ++i) {
      scf::IfOp consumer = depEntry.consumers[i];
      int argIdx = argIndices[i];
      if (argIdx >= 0 && argIdx < static_cast<int>(regionIterArgs.size())) {
        consumerToVar[consumer] = regionIterArgs[argIdx];
      }
    }

    if (depEntry.producer) {
      for (auto &[consumer, var] : consumerToVar)
        producerVars[depEntry.producer].insert(var);
    }
    for (auto &[consumer, var] : consumerToVar)
      consumerVars[consumer].insert(var);
  }

  for (auto &[producer, vars] : producerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[producer];
    for (Value var : vars)
      ifOpVars.producerVars.push_back(var);
  }
  for (auto &[consumer, vars] : consumerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[consumer];
    for (Value var : vars)
      ifOpVars.consumerVars.push_back(var);
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar,
    SmallVector<OutputGroupInfo> &outputGroups) {
  outputGroups.clear();
  for (int idx : intraCoreOutputValues) {
    auto bufferIt = intraCoreBuffers.find(idx);
    if (bufferIt == intraCoreBuffers.end())
      continue;
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end())
      continue;
    Value var = varIt->second;

    for (auto &entry : bufferIt->second) {
      SmallVector<Operation *> outputOps;
      for (Operation *producer : entry.second)
        outputOps.push_back(producer);
      if (outputOps.empty())
        continue;

      bool flag = true;
      for (auto &outputGroup : outputGroups) {
        if (outputGroup.outputs == outputOps) {
          outputGroup.inputVars.push_back(var);
          flag = false;
          break;
        }
      }
      if (flag) {
        OutputGroupInfo groupInfo;
        groupInfo.outputs = outputOps;
        groupInfo.inputVars.push_back(var);
        outputGroups.push_back(groupInfo);
      }
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

void UpdateConditionInfoPass::collectIntraCoreInputConditions(
    OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
    DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
    DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  for (int idx : intraCoreInputValues) {
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end())
      continue;
    Value var = varIt->second;
    Value varToUse = getLatestValue(controlVarToLatestValue, var);

    Value zeroConst = createIntConst(builder, loc, 0, varToUse.getType());
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               varToUse, zeroConst)
                     .getResult();
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
  }
}

int UpdateConditionInfoPass::collectIntraCoreOutputConditions(
    OpBuilder &builder, Location loc,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  SmallVector<OutputGroupInfo> outputGroups;
  if (buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar,
                        outputGroups) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  for (auto &group : outputGroups) {
    int size = group.outputs.size();
    for (Value var : group.inputVars) {
      Value varToUse = getLatestValue(controlVarToLatestValue, var);
      Value limitVal = createIntConst(builder, loc, size, varToUse.getType());
      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 varToUse, limitVal)
                       .getResult();
      conditions.push_back(cond);
      usedVarsSet.insert(var);
      varUpdateTypes[var] = VarUpdateType::INC;
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

void UpdateConditionInfoPass::collectTensorIterArgInputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp))
    return;
  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.consumerVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, var);
    Value oneConst = createIntConst(builder, loc, 1, varToUse.getType());
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, oneConst)
                     .getResult();
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
  }
}

void UpdateConditionInfoPass::collectTensorIterArgOutputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp))
    return;
  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.producerVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, var);
    Value zeroConst = createIntConst(builder, loc, 0, varToUse.getType());
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, zeroConst)
                     .getResult();
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::INC;
  }
}

// Cross-core conditions using scalar iter_arg tokens (No SSBuffer!)
void UpdateConditionInfoPass::collectCrossCoreTokenConditions(
    OpBuilder &builder, Location loc,
    const SmallVector<int> &crossCoreInputValues,
    const SmallVector<int> &crossCoreOutputValues,
    const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &crossCoreBuffers,
    const DenseMap<int, Value> &crossCoreTokenMap,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  for (int inputIdx : crossCoreInputValues) {
    auto it = crossCoreTokenMap.find(inputIdx);
    if (it == crossCoreTokenMap.end())
      continue;
    Value token = it->second;
    Value tokenToUse = getLatestValue(controlVarToLatestValue, token);

    Value zero = createIntConst(builder, loc, 0, tokenToUse.getType());
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               tokenToUse, zero)
                     .getResult();
    conditions.push_back(cond);
    usedVarsSet.insert(token);
    varUpdateTypes[token] = VarUpdateType::DEC;
  }

  for (int outputIdx : crossCoreOutputValues) {
    auto it = crossCoreTokenMap.find(outputIdx);
    if (it == crossCoreTokenMap.end())
      continue;
    Value token = it->second;
    Value tokenToUse = getLatestValue(controlVarToLatestValue, token);

    int outputCount = 0;
    auto bufIt = crossCoreBuffers.find(outputIdx);
    if (bufIt != crossCoreBuffers.end()) {
      for (auto &entry : bufIt->second)
        outputCount += entry.second.size();
    }
    if (outputCount == 0)
      outputCount = 1;

    Value limitVal =
        createIntConst(builder, loc, outputCount, tokenToUse.getType());
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               tokenToUse, limitVal)
                     .getResult();
    conditions.push_back(cond);
    usedVarsSet.insert(token);
    varUpdateTypes[token] = VarUpdateType::INC;
  }
}

int UpdateConditionInfoPass::setFlowOptCondition(scf::IfOp currentIfOp,
                                                 Operation *loopOp,
                                                 Value counter,
                                                 Value &flowOptCond) {
  flowOptCond = nullptr;
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp)
    return UPDATE_CONDITION_INFO_SUCCESS;

  if (!info->flowOptIfOpPairs.count(currentIfOp))
    return UPDATE_CONDITION_INFO_SUCCESS;

  if (info->crossCoreBufferCount <= CROSS_CORE_BUFFER_COUNT_THRESHOLD ||
      info->intraCoreBufferCount <= INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  scf::IfOp sourceIfOp = info->flowOptIfOpPairs[currentIfOp];
  if (!info->cntArgs.count(sourceIfOp)) {
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  Value srcCounter = info->cntArgs[sourceIfOp];
  Value srcCounterToUse =
      getLatestValue(controlVarToLatestValue, srcCounter);

  OpBuilder builder(currentIfOp);
  Location loc = currentIfOp.getLoc();

  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              srcCounterToUse, upperBound)
                    .getResult();

  int optInt =
      std::min(info->intraCoreBufferCount - 1, info->crossCoreBufferCount);
  auto stepIntType = dyn_cast<IntegerType>(step.getType());
  if (!stepIntType)
    return UPDATE_CONDITION_INFO_FAILED;

  Value optNum = builder.create<arith::ConstantIntOp>(
                            loc, optInt, stepIntType.getWidth())
                     .getResult();
  Value optOffset =
      builder.create<arith::MulIOp>(loc, step, optNum).getResult();
  Value lowerPlusOffset =
      builder.create<arith::AddIOp>(loc, lowerBound, optOffset).getResult();
  Value cond2 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              srcCounterToUse, lowerPlusOffset)
                    .getResult();

  flowOptCond = builder.create<arith::OrIOp>(loc, cond1, cond2).getResult();
  return UPDATE_CONDITION_INFO_SUCCESS;
}

scf::IfOp UpdateConditionInfoPass::createRealIfOp(
    scf::IfOp oldIfOp, Value combinedCond, const SmallVector<Value> &usedVars,
    const DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  OpBuilder builder(oldIfOp);
  Location loc = oldIfOp.getLoc();

  SmallVector<Type> resultTypes(oldIfOp.getResultTypes().begin(),
                                oldIfOp.getResultTypes().end());
  for (Value v : usedVars)
    resultTypes.push_back(v.getType());
  if (hasCounter)
    resultTypes.push_back(counter.getType());

  scf::IfOp newIfOp = builder.create<scf::IfOp>(loc, resultTypes, combinedCond,
                                                /*withElse=*/true);

  for (auto &attr : oldIfOp->getAttrs())
    newIfOp->setAttr(attr.getName(), attr.getValue());

  // Populate Then Block
  Block *newThenBlock = newIfOp.thenBlock();
  Block *oldThenBlock = oldIfOp.thenBlock();
  auto oldThenYield = cast<scf::YieldOp>(oldThenBlock->getTerminator());
  SmallVector<Value> oldYieldOperands(oldThenYield.getOperands().begin(),
                                      oldThenYield.getOperands().end());

  newThenBlock->getOperations().splice(
      newThenBlock->end(), oldThenBlock->getOperations(), oldThenBlock->begin(),
      Block::iterator(oldThenYield));

  builder.setInsertionPointToEnd(newThenBlock);
  SmallVector<Value> thenYieldOperands = oldYieldOperands;

  for (Value origVar : usedVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, origVar);
    auto it = varUpdateTypes.find(origVar);
    VarUpdateType updateType = (it != varUpdateTypes.end())
                                   ? it->second
                                   : VarUpdateType::INC;
    Value one = createIntConst(builder, loc, 1, varToUse.getType());
    Value updated = (updateType == VarUpdateType::INC)
                        ? builder.create<arith::AddIOp>(loc, varToUse, one)
                              .getResult()
                        : builder.create<arith::SubIOp>(loc, varToUse, one)
                              .getResult();
    thenYieldOperands.push_back(updated);
  }

  if (hasCounter) {
    Value counterToUse = getLatestValue(controlVarToLatestValue, counter);
    Value updatedCounter =
        builder.create<arith::AddIOp>(loc, counterToUse, step).getResult();
    thenYieldOperands.push_back(updatedCounter);
  }

  builder.create<scf::YieldOp>(loc, thenYieldOperands);

  // Populate Else Block
  Block *newElseBlock = newIfOp.elseBlock();
  builder.setInsertionPointToEnd(newElseBlock);
  SmallVector<Value> elseYieldOperands;

  if (oldIfOp.elseBlock()) {
    Block *oldElseBlock = oldIfOp.elseBlock();
    auto oldElseYield = cast<scf::YieldOp>(oldElseBlock->getTerminator());
    elseYieldOperands.assign(oldElseYield.getOperands().begin(),
                             oldElseYield.getOperands().end());
    newElseBlock->getOperations().splice(
        newElseBlock->end(), oldElseBlock->getOperations(),
        oldElseBlock->begin(), Block::iterator(oldElseYield));
    builder.setInsertionPointToEnd(newElseBlock);
  }

  // Forward unmodified control variables in else block
  for (Value origVar : usedVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, origVar);
    elseYieldOperands.push_back(varToUse);
  }
  if (hasCounter) {
    Value counterToUse = getLatestValue(controlVarToLatestValue, counter);
    elseYieldOperands.push_back(counterToUse);
  }

  builder.create<scf::YieldOp>(loc, elseYieldOperands);

  return newIfOp;
}

scf::IfOp UpdateConditionInfoPass::createDummyIfOp(
    OpBuilder &builder, Location loc, int blockId, Value combinedCond,
    const SmallVector<Value> &usedVars,
    const DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  SmallVector<Type> resultTypes;
  for (Value v : usedVars)
    resultTypes.push_back(v.getType());
  if (hasCounter)
    resultTypes.push_back(counter.getType());

  scf::IfOp newIfOp = builder.create<scf::IfOp>(loc, resultTypes, combinedCond,
                                                /*withElse=*/true);
  newIfOp->setAttr(CVPipeline::kIf, builder.getI32IntegerAttr(blockId));

  // Then block: empty compute body, update usedVars and counter
  Block *thenBlock = newIfOp.thenBlock();
  OpBuilder thenBuilder(thenBlock, thenBlock->end());
  SmallVector<Value> thenYieldOperands;

  for (Value origVar : usedVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, origVar);
    auto it = varUpdateTypes.find(origVar);
    VarUpdateType updateType = (it != varUpdateTypes.end())
                                   ? it->second
                                   : VarUpdateType::INC;
    Value one = createIntConst(thenBuilder, loc, 1, varToUse.getType());
    Value updated = (updateType == VarUpdateType::INC)
                        ? thenBuilder.create<arith::AddIOp>(loc, varToUse, one)
                              .getResult()
                        : thenBuilder.create<arith::SubIOp>(loc, varToUse, one)
                              .getResult();
    thenYieldOperands.push_back(updated);
  }

  if (hasCounter) {
    Value counterToUse = getLatestValue(controlVarToLatestValue, counter);
    Value updatedCounter =
        thenBuilder.create<arith::AddIOp>(loc, counterToUse, step).getResult();
    thenYieldOperands.push_back(updatedCounter);
  }

  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);

  // Else block: forward unmodified values
  Block *elseBlock = newIfOp.elseBlock();
  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  SmallVector<Value> elseYieldOperands;

  for (Value origVar : usedVars) {
    Value varToUse = getLatestValue(controlVarToLatestValue, origVar);
    elseYieldOperands.push_back(varToUse);
  }
  if (hasCounter) {
    Value counterToUse = getLatestValue(controlVarToLatestValue, counter);
    elseYieldOperands.push_back(counterToUse);
  }

  elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);

  return newIfOp;
}

void UpdateConditionInfoPass::updateDAGAfterIfOpReplacement(scf::IfOp oldIfOp,
                                                            scf::IfOp newIfOp) {
  if (info->ifBlockDAG.count(oldIfOp)) {
    auto consumers = info->ifBlockDAG[oldIfOp];
    info->ifBlockDAG.erase(oldIfOp);
    info->ifBlockDAG[newIfOp] = consumers;
  }
  for (auto &entry : info->ifBlockDAG) {
    for (auto &edge : entry.second) {
      if (edge.first == oldIfOp)
        edge.first = newIfOp;
    }
  }

  if (info->flowOptIfOpPairs.count(oldIfOp)) {
    auto source = info->flowOptIfOpPairs[oldIfOp];
    info->flowOptIfOpPairs.erase(oldIfOp);
    info->flowOptIfOpPairs[newIfOp] = source;
  }
  for (auto &entry : info->flowOptIfOpPairs) {
    if (entry.second == oldIfOp)
      entry.second = newIfOp;
  }
}

void UpdateConditionInfoPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  LDBG("Enter UpdateConditionInfoPass (No-SSBuffer Lockstep Architecture)");

  Operation *cubeScope = nullptr;
  Operation *vectorScope = nullptr;
  scf::ForOp cubeForOp = nullptr;
  scf::ForOp vectorForOp = nullptr;

  if (failed(findCubeAndVectorLoops(module, cubeScope, cubeForOp, vectorScope,
                                    vectorForOp))) {
    LDBG("Failed to find loops, skipping pass");
    return;
  }

  SmallVector<Operation *> mainLoops;
  if (cubeForOp)
    mainLoops.push_back(cubeForOp);
  if (vectorForOp && vectorForOp != cubeForOp)
    mainLoops.push_back(vectorForOp);

  if (mainLoops.empty()) {
    LDBG("No main loops found");
    return;
  }

  info->cntArgs.clear();

  DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
      crossCoreBuffers;
  DenseMap<Operation *,
           DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
      intraCoreBuffersMap;
  collectDependencyBuffers(module, mainLoops, crossCoreBuffers,
                           intraCoreBuffersMap);

  size_t numCrossCore = crossCoreBuffers.size();
  LDBG("Collected " << numCrossCore << " cross-core buffer groups");

  SmallVector<scf::IfOp> rawCubeIfOps;
  SmallVector<scf::IfOp> rawVectorIfOps;
  if (cubeForOp)
    collectSSBufferIfOps(cubeForOp, rawCubeIfOps);
  if (vectorForOp)
    collectSSBufferIfOps(vectorForOp, rawVectorIfOps);

  std::vector<IfOpInfo> cubeIfInfos;
  for (size_t i = 0; i < rawCubeIfOps.size(); ++i) {
    scf::IfOp ifOp = rawCubeIfOps[i];
    IfOpInfo infoOp;
    infoOp.ifOp = ifOp;
    infoOp.isAIC = true;
    if (getIfBlockId(ifOp, infoOp.blockId) != UPDATE_CONDITION_INFO_SUCCESS)
      continue;
    getInputOutputValues(ifOp, crossCoreBuffers, intraCoreBuffersMap[cubeForOp],
                         infoOp.crossCoreInputValues,
                         infoOp.crossCoreOutputValues,
                         infoOp.intraCoreInputValues,
                         infoOp.intraCoreOutputValues);
    cubeIfInfos.push_back(infoOp);
  }

  std::vector<IfOpInfo> vectorIfInfos;
  for (size_t i = 0; i < rawVectorIfOps.size(); ++i) {
    scf::IfOp ifOp = rawVectorIfOps[i];
    IfOpInfo infoOp;
    infoOp.ifOp = ifOp;
    infoOp.isAIC = false;
    if (getIfBlockId(ifOp, infoOp.blockId) != UPDATE_CONDITION_INFO_SUCCESS)
      continue;
    getInputOutputValues(ifOp, crossCoreBuffers,
                         intraCoreBuffersMap[vectorForOp],
                         infoOp.crossCoreInputValues,
                         infoOp.crossCoreOutputValues,
                         infoOp.intraCoreInputValues,
                         infoOp.intraCoreOutputValues);
    vectorIfInfos.push_back(infoOp);
  }

  // Rebuild Cube loop with extra iter_args
  scf::ForOp newCubeForOp = cubeForOp;
  unsigned oldCubeNumArgs = cubeForOp ? cubeForOp.getNumRegionIterArgs() : 0;
  if (cubeForOp && (numCrossCore > 0 || !vectorIfInfos.empty())) {
    OpBuilder b(cubeForOp);
    Location loc = cubeForOp.getLoc();
    auto i32Type = b.getI32Type();
    SmallVector<Value> extraInitArgs;
    for (size_t i = 0; i < numCrossCore; ++i) {
      extraInitArgs.push_back(createIntConst(b, loc, 0, i32Type));
    }
    for (size_t i = 0; i < vectorIfInfos.size(); ++i) {
      extraInitArgs.push_back(cubeForOp.getLowerBound());
    }
    newCubeForOp = createNewForOpWithExtras(cubeForOp, extraInitArgs);
    migrateBody(cubeForOp.getBody(), newCubeForOp.getBody());
    SmallVector<Value> extraYieldValues;
    for (size_t i = 0; i < extraInitArgs.size(); ++i) {
      extraYieldValues.push_back(
          newCubeForOp.getBody()->getArgument(1 + oldCubeNumArgs + i));
    }
    (void)buildNewYieldOp(cubeForOp.getBody(), newCubeForOp.getBody(),
                          newCubeForOp, extraYieldValues);
    replaceOpResultUses(cubeForOp, newCubeForOp);

    if (info->blockCounters.count(cubeForOp)) {
      info->blockCounters[newCubeForOp] = info->blockCounters[cubeForOp];
      info->blockCounters.erase(cubeForOp);
    }
    if (info->innerDepConds.count(cubeForOp)) {
      info->innerDepConds[newCubeForOp] = info->innerDepConds[cubeForOp];
      info->innerDepConds.erase(cubeForOp);
    }
    if (info->tensorIterArgDepsMap.count(cubeForOp)) {
      info->tensorIterArgDepsMap[newCubeForOp] =
          info->tensorIterArgDepsMap[cubeForOp];
      info->tensorIterArgDepsMap.erase(cubeForOp);
    }
    if (info->tensorIterArgIndicesMap.count(cubeForOp)) {
      info->tensorIterArgIndicesMap[newCubeForOp] =
          info->tensorIterArgIndicesMap[cubeForOp];
      info->tensorIterArgIndicesMap.erase(cubeForOp);
    }
    cubeForOp.erase();
    cubeForOp = newCubeForOp;
  }

  // Rebuild Vector loop with extra iter_args
  scf::ForOp newVectorForOp = vectorForOp;
  unsigned oldVectorNumArgs =
      vectorForOp ? vectorForOp.getNumRegionIterArgs() : 0;
  if (vectorForOp && (numCrossCore > 0 || !cubeIfInfos.empty())) {
    OpBuilder b(vectorForOp);
    Location loc = vectorForOp.getLoc();
    auto i32Type = b.getI32Type();
    SmallVector<Value> extraInitArgs;
    for (size_t i = 0; i < numCrossCore; ++i) {
      extraInitArgs.push_back(createIntConst(b, loc, 0, i32Type));
    }
    for (size_t i = 0; i < cubeIfInfos.size(); ++i) {
      extraInitArgs.push_back(vectorForOp.getLowerBound());
    }
    newVectorForOp = createNewForOpWithExtras(vectorForOp, extraInitArgs);
    migrateBody(vectorForOp.getBody(), newVectorForOp.getBody());
    SmallVector<Value> extraYieldValues;
    for (size_t i = 0; i < extraInitArgs.size(); ++i) {
      extraYieldValues.push_back(
          newVectorForOp.getBody()->getArgument(1 + oldVectorNumArgs + i));
    }
    (void)buildNewYieldOp(vectorForOp.getBody(), newVectorForOp.getBody(),
                          newVectorForOp, extraYieldValues);
    replaceOpResultUses(vectorForOp, newVectorForOp);

    if (info->blockCounters.count(vectorForOp)) {
      info->blockCounters[newVectorForOp] = info->blockCounters[vectorForOp];
      info->blockCounters.erase(vectorForOp);
    }
    if (info->innerDepConds.count(vectorForOp)) {
      info->innerDepConds[newVectorForOp] = info->innerDepConds[vectorForOp];
      info->innerDepConds.erase(vectorForOp);
    }
    if (info->tensorIterArgDepsMap.count(vectorForOp)) {
      info->tensorIterArgDepsMap[newVectorForOp] =
          info->tensorIterArgDepsMap[vectorForOp];
      info->tensorIterArgDepsMap.erase(vectorForOp);
    }
    if (info->tensorIterArgIndicesMap.count(vectorForOp)) {
      info->tensorIterArgIndicesMap[newVectorForOp] =
          info->tensorIterArgIndicesMap[vectorForOp];
      info->tensorIterArgIndicesMap.erase(vectorForOp);
    }
    vectorForOp.erase();
    vectorForOp = newVectorForOp;
  }

  // Transform Cube loop body: [Real Cube IfOps] -> [Dummy Vector IfOps]
  if (newCubeForOp) {
    controlVarToLatestValue.clear();
    DenseMap<int, Value> crossCoreTokenMap;
    for (size_t g = 0; g < numCrossCore; ++g) {
      crossCoreTokenMap[g] =
          newCubeForOp.getRegionIterArgs()[oldCubeNumArgs + g];
    }
    SmallVector<Value> shadowVectorCounters;
    for (size_t j = 0; j < vectorIfInfos.size(); ++j) {
      shadowVectorCounters.push_back(
          newCubeForOp.getRegionIterArgs()[oldCubeNumArgs + numCrossCore + j]);
    }

    DenseMap<int, Value> idxToVar;
    (void)buildIdxToVarMap(newCubeForOp, intraCoreBuffersMap[newCubeForOp],
                           idxToVar);
    (void)buildTensorIterArgIfOpVarMap(newCubeForOp);

    Value upperBound = newCubeForOp.getUpperBound();
    Value step = newCubeForOp.getStep();
    Location loc = newCubeForOp.getLoc();

    // 1. Process Real Cube IfOps
    for (size_t i = 0; i < cubeIfInfos.size(); ++i) {
      auto &infoOp = cubeIfInfos[i];
      scf::IfOp ifOp = infoOp.ifOp;
      OpBuilder builder(ifOp);

      Value counter = nullptr;
      bool hasCounter = false;
      if (info->blockCounters.count(newCubeForOp) &&
          i < info->blockCounters[newCubeForOp].size()) {
        int argIdx = info->blockCounters[newCubeForOp][i];
        counter = newCubeForOp.getRegionIterArgs()[argIdx];
        hasCounter = true;
      }

      SmallVector<Value> conditions;
      DenseSet<Value> usedVarsSet;
      DenseMap<Value, VarUpdateType> varUpdateTypes;

      collectIntraCoreInputConditions(builder, ifOp.getLoc(),
                                      infoOp.intraCoreInputValues, idxToVar,
                                      conditions, usedVarsSet, varUpdateTypes);
      (void)collectIntraCoreOutputConditions(
          builder, ifOp.getLoc(), intraCoreBuffersMap[newCubeForOp],
          infoOp.intraCoreOutputValues, idxToVar, conditions, usedVarsSet,
          varUpdateTypes);
      collectTensorIterArgInputConditions(builder, ifOp.getLoc(), ifOp,
                                          conditions, usedVarsSet,
                                          varUpdateTypes);
      collectTensorIterArgOutputConditions(builder, ifOp.getLoc(), ifOp,
                                           conditions, usedVarsSet,
                                           varUpdateTypes);
      collectCrossCoreTokenConditions(
          builder, ifOp.getLoc(), infoOp.crossCoreInputValues,
          infoOp.crossCoreOutputValues, crossCoreBuffers, crossCoreTokenMap,
          conditions, usedVarsSet, varUpdateTypes);

      if (hasCounter) {
        Value counterToUse =
            getLatestValue(controlVarToLatestValue, counter);
        Value counterCond = builder.create<arith::CmpIOp>(
                                       ifOp.getLoc(),
                                       arith::CmpIPredicate::slt, counterToUse,
                                       upperBound)
                                .getResult();
        conditions.push_back(counterCond);
      }

      Value flowOptCond = nullptr;
      (void)setFlowOptCondition(ifOp, newCubeForOp, counter, flowOptCond);
      if (flowOptCond)
        conditions.push_back(flowOptCond);

      Value combinedCond;
      if (conditions.empty()) {
        combinedCond =
            createIntConst(builder, ifOp.getLoc(), 1, builder.getI1Type());
      } else {
        combinedCond = conditions.front();
        for (size_t c = 1; c < conditions.size(); ++c) {
          combinedCond = builder.create<arith::AndIOp>(
                                    ifOp.getLoc(), combinedCond, conditions[c])
                             .getResult();
        }
      }

      SmallVector<Value> usedVars(usedVarsSet.begin(), usedVarsSet.end());
      scf::IfOp newRealIfOp =
          createRealIfOp(ifOp, combinedCond, usedVars, varUpdateTypes,
                         hasCounter, counter, step);

      size_t origNumRes = ifOp.getNumResults();
      for (size_t u = 0; u < usedVars.size(); ++u) {
        controlVarToLatestValue[usedVars[u]] =
            newRealIfOp.getResult(origNumRes + u);
      }
      if (hasCounter) {
        controlVarToLatestValue[counter] =
            newRealIfOp.getResult(origNumRes + usedVars.size());
        info->cntArgs[newRealIfOp] = counter;
      }

      updateDAGAfterIfOpReplacement(ifOp, newRealIfOp);
      if (info->tensorIterArgDepsMap.count(newCubeForOp)) {
        for (auto &dep : info->tensorIterArgDepsMap[newCubeForOp]) {
          if (dep.producer == ifOp)
            dep.producer = newRealIfOp;
          for (auto &cons : dep.consumers) {
            if (cons == ifOp)
              cons = newRealIfOp;
          }
        }
      }

      ifOp.replaceAllUsesWith(newRealIfOp.getResults().take_front(origNumRes));
      ifOp.erase();
    }

    // 2. Insert Dummy Vector IfOps before terminator
    Operation *terminator = newCubeForOp.getBody()->getTerminator();
    OpBuilder dummyBuilder(terminator);
    for (size_t j = 0; j < vectorIfInfos.size(); ++j) {
      auto &infoOp = vectorIfInfos[j];
      Value shadowCounter = shadowVectorCounters[j];
      bool hasCounter = (shadowCounter != nullptr);

      SmallVector<Value> conditions;
      DenseSet<Value> usedVarsSet;
      DenseMap<Value, VarUpdateType> varUpdateTypes;

      collectCrossCoreTokenConditions(
          dummyBuilder, loc, infoOp.crossCoreInputValues,
          infoOp.crossCoreOutputValues, crossCoreBuffers, crossCoreTokenMap,
          conditions, usedVarsSet, varUpdateTypes);

      if (hasCounter) {
        Value counterToUse =
            getLatestValue(controlVarToLatestValue, shadowCounter);
        Value counterCond = dummyBuilder.create<arith::CmpIOp>(
                                            loc, arith::CmpIPredicate::slt,
                                            counterToUse, upperBound)
                                .getResult();
        conditions.push_back(counterCond);
      }

      Value combinedCond;
      if (conditions.empty()) {
        combinedCond =
            createIntConst(dummyBuilder, loc, 1, dummyBuilder.getI1Type());
      } else {
        combinedCond = conditions.front();
        for (size_t c = 1; c < conditions.size(); ++c) {
          combinedCond = dummyBuilder.create<arith::AndIOp>(
                                         loc, combinedCond, conditions[c])
                             .getResult();
        }
      }

      SmallVector<Value> usedVars(usedVarsSet.begin(), usedVarsSet.end());
      scf::IfOp newDummyIfOp =
          createDummyIfOp(dummyBuilder, loc, infoOp.blockId, combinedCond,
                          usedVars, varUpdateTypes, hasCounter, shadowCounter,
                          step);

      for (size_t u = 0; u < usedVars.size(); ++u) {
        controlVarToLatestValue[usedVars[u]] = newDummyIfOp.getResult(u);
      }
      if (hasCounter) {
        controlVarToLatestValue[shadowCounter] =
            newDummyIfOp.getResult(usedVars.size());
        info->cntArgs[newDummyIfOp] = shadowCounter;
      }
    }

    // 3. Update Cube loop yield terminator
    auto yieldOp = cast<scf::YieldOp>(newCubeForOp.getBody()->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (unsigned a = 0; a < newCubeForOp.getNumRegionIterArgs(); ++a) {
      Value iterArg = newCubeForOp.getRegionIterArgs()[a];
      Value latestVal = getLatestValue(controlVarToLatestValue, iterArg);
      newYieldOperands.push_back(latestVal);
    }
    OpBuilder yieldBuilder(yieldOp);
    yieldBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    yieldOp.erase();
  }

  // Transform Vector loop body: [Dummy Cube IfOps] -> [Real Vector IfOps]
  if (newVectorForOp) {
    controlVarToLatestValue.clear();
    DenseMap<int, Value> crossCoreTokenMap;
    for (size_t g = 0; g < numCrossCore; ++g) {
      crossCoreTokenMap[g] =
          newVectorForOp.getRegionIterArgs()[oldVectorNumArgs + g];
    }
    SmallVector<Value> shadowCubeCounters;
    for (size_t i = 0; i < cubeIfInfos.size(); ++i) {
      shadowCubeCounters.push_back(
          newVectorForOp.getRegionIterArgs()[oldVectorNumArgs + numCrossCore + i]);
    }

    DenseMap<int, Value> idxToVar;
    (void)buildIdxToVarMap(newVectorForOp, intraCoreBuffersMap[newVectorForOp],
                           idxToVar);
    (void)buildTensorIterArgIfOpVarMap(newVectorForOp);

    Value upperBound = newVectorForOp.getUpperBound();
    Value step = newVectorForOp.getStep();
    Location loc = newVectorForOp.getLoc();

    // 1. Insert Dummy Cube IfOps at the start of Vector loop body
    Operation *firstOp = &newVectorForOp.getBody()->front();
    OpBuilder dummyBuilder(firstOp);
    for (size_t i = 0; i < cubeIfInfos.size(); ++i) {
      auto &infoOp = cubeIfInfos[i];
      Value shadowCounter = shadowCubeCounters[i];
      bool hasCounter = (shadowCounter != nullptr);

      SmallVector<Value> conditions;
      DenseSet<Value> usedVarsSet;
      DenseMap<Value, VarUpdateType> varUpdateTypes;

      collectCrossCoreTokenConditions(
          dummyBuilder, loc, infoOp.crossCoreInputValues,
          infoOp.crossCoreOutputValues, crossCoreBuffers, crossCoreTokenMap,
          conditions, usedVarsSet, varUpdateTypes);

      if (hasCounter) {
        Value counterToUse =
            getLatestValue(controlVarToLatestValue, shadowCounter);
        Value counterCond = dummyBuilder.create<arith::CmpIOp>(
                                            loc, arith::CmpIPredicate::slt,
                                            counterToUse, upperBound)
                                .getResult();
        conditions.push_back(counterCond);
      }

      Value combinedCond;
      if (conditions.empty()) {
        combinedCond =
            createIntConst(dummyBuilder, loc, 1, dummyBuilder.getI1Type());
      } else {
        combinedCond = conditions.front();
        for (size_t c = 1; c < conditions.size(); ++c) {
          combinedCond = dummyBuilder.create<arith::AndIOp>(
                                         loc, combinedCond, conditions[c])
                             .getResult();
        }
      }

      SmallVector<Value> usedVars(usedVarsSet.begin(), usedVarsSet.end());
      scf::IfOp newDummyIfOp =
          createDummyIfOp(dummyBuilder, loc, infoOp.blockId, combinedCond,
                          usedVars, varUpdateTypes, hasCounter, shadowCounter,
                          step);

      for (size_t u = 0; u < usedVars.size(); ++u) {
        controlVarToLatestValue[usedVars[u]] = newDummyIfOp.getResult(u);
      }
      if (hasCounter) {
        controlVarToLatestValue[shadowCounter] =
            newDummyIfOp.getResult(usedVars.size());
        info->cntArgs[newDummyIfOp] = shadowCounter;
      }
    }

    // 2. Process Real Vector IfOps
    for (size_t j = 0; j < vectorIfInfos.size(); ++j) {
      auto &infoOp = vectorIfInfos[j];
      scf::IfOp ifOp = infoOp.ifOp;
      OpBuilder builder(ifOp);

      Value counter = nullptr;
      bool hasCounter = false;
      if (info->blockCounters.count(newVectorForOp) &&
          j < info->blockCounters[newVectorForOp].size()) {
        int argIdx = info->blockCounters[newVectorForOp][j];
        counter = newVectorForOp.getRegionIterArgs()[argIdx];
        hasCounter = true;
      }

      SmallVector<Value> conditions;
      DenseSet<Value> usedVarsSet;
      DenseMap<Value, VarUpdateType> varUpdateTypes;

      collectIntraCoreInputConditions(builder, ifOp.getLoc(),
                                      infoOp.intraCoreInputValues, idxToVar,
                                      conditions, usedVarsSet, varUpdateTypes);
      (void)collectIntraCoreOutputConditions(
          builder, ifOp.getLoc(), intraCoreBuffersMap[newVectorForOp],
          infoOp.intraCoreOutputValues, idxToVar, conditions, usedVarsSet,
          varUpdateTypes);
      collectTensorIterArgInputConditions(builder, ifOp.getLoc(), ifOp,
                                          conditions, usedVarsSet,
                                          varUpdateTypes);
      collectTensorIterArgOutputConditions(builder, ifOp.getLoc(), ifOp,
                                           conditions, usedVarsSet,
                                           varUpdateTypes);
      collectCrossCoreTokenConditions(
          builder, ifOp.getLoc(), infoOp.crossCoreInputValues,
          infoOp.crossCoreOutputValues, crossCoreBuffers, crossCoreTokenMap,
          conditions, usedVarsSet, varUpdateTypes);

      if (hasCounter) {
        Value counterToUse =
            getLatestValue(controlVarToLatestValue, counter);
        Value counterCond = builder.create<arith::CmpIOp>(
                                       ifOp.getLoc(),
                                       arith::CmpIPredicate::slt, counterToUse,
                                       upperBound)
                                .getResult();
        conditions.push_back(counterCond);
      }

      Value flowOptCond = nullptr;
      (void)setFlowOptCondition(ifOp, newVectorForOp, counter, flowOptCond);
      if (flowOptCond)
        conditions.push_back(flowOptCond);

      Value combinedCond;
      if (conditions.empty()) {
        combinedCond =
            createIntConst(builder, ifOp.getLoc(), 1, builder.getI1Type());
      } else {
        combinedCond = conditions.front();
        for (size_t c = 1; c < conditions.size(); ++c) {
          combinedCond = builder.create<arith::AndIOp>(
                                    ifOp.getLoc(), combinedCond, conditions[c])
                             .getResult();
        }
      }

      SmallVector<Value> usedVars(usedVarsSet.begin(), usedVarsSet.end());
      scf::IfOp newRealIfOp =
          createRealIfOp(ifOp, combinedCond, usedVars, varUpdateTypes,
                         hasCounter, counter, step);

      size_t origNumRes = ifOp.getNumResults();
      for (size_t u = 0; u < usedVars.size(); ++u) {
        controlVarToLatestValue[usedVars[u]] =
            newRealIfOp.getResult(origNumRes + u);
      }
      if (hasCounter) {
        controlVarToLatestValue[counter] =
            newRealIfOp.getResult(origNumRes + usedVars.size());
        info->cntArgs[newRealIfOp] = counter;
      }

      updateDAGAfterIfOpReplacement(ifOp, newRealIfOp);
      if (info->tensorIterArgDepsMap.count(newVectorForOp)) {
        for (auto &dep : info->tensorIterArgDepsMap[newVectorForOp]) {
          if (dep.producer == ifOp)
            dep.producer = newRealIfOp;
          for (auto &cons : dep.consumers) {
            if (cons == ifOp)
              cons = newRealIfOp;
          }
        }
      }

      ifOp.replaceAllUsesWith(newRealIfOp.getResults().take_front(origNumRes));
      ifOp.erase();
    }

    // 3. Update Vector loop yield terminator
    auto yieldOp =
        cast<scf::YieldOp>(newVectorForOp.getBody()->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (unsigned a = 0; a < newVectorForOp.getNumRegionIterArgs(); ++a) {
      Value iterArg = newVectorForOp.getRegionIterArgs()[a];
      Value latestVal = getLatestValue(controlVarToLatestValue, iterArg);
      newYieldOperands.push_back(latestVal);
    }
    OpBuilder yieldBuilder(yieldOp);
    yieldBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    yieldOp.erase();
  }

  LDBG("Exit UpdateConditionInfoPass successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass() {
  return std::make_unique<UpdateConditionInfoPass>();
}

} // namespace triton
} // namespace mlir
