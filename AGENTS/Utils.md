# Dynamic CV Pipeline Utilities & Helper Functions (Utils.md)

This document indexes reusable auxiliary functions and helpers across `DynamicCVPipeline`, noting their current implementation locations for easy reuse or future refactoring into shared headers.

---

## 1. Loop Construction & Migration Helpers
Located in:
- Header: `third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h`
- Implementation: `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/Utils.cpp`

### `createNewForOpWithExtras`
```cpp
scf::ForOp createNewForOpWithExtras(scf::ForOp oldForOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:363`
- **Purpose**: Creates a new `scf.for` with matching bounds/step/attributes, with `extraInitArgs` appended to the end of original `initArgs`. Returns `oldForOp` unchanged if `extraInitArgs` is empty.

### `createNewWhileOpWithExtras`
```cpp
scf::WhileOp createNewWhileOpWithExtras(scf::WhileOp oldWhileOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:386`
- **Purpose**: Creates a new `scf.while` with extra inits appended, allocating empty before and after blocks.

### `createMainLoopOpWithExtras`
```cpp
Operation *createMainLoopOpWithExtras(Operation *oldOp, ArrayRef<Value> extraInitArgs);
```
- **Location**: `Utils.cpp:426`
- **Purpose**: Polymorphic dispatcher for `createNewForOpWithExtras` or `createNewWhileOpWithExtras`.

### `migrateBody`
```cpp
void migrateBody(Block *oldBlock, Block *newBlock);
```
- **Location**: `Utils.cpp:294`
- **Purpose**: Rewires `oldBlock` arguments to `newBlock` matching arguments via `replaceAllUsesWith`, and splices all non-terminator operations into `newBlock`.

### `migrateWhileBodies`
```cpp
void migrateWhileBodies(scf::WhileOp oldWhileOp, scf::WhileOp newWhileOp);
```
- **Location**: `Utils.cpp:307`
- **Purpose**: Calls `migrateBody` on both `before` and `after` regions of an `scf.while`.

### `buildNewYieldOp`
```cpp
LogicalResult buildNewYieldOp(Block *oldBlock, Block *newBlock, Operation *newOp, ArrayRef<Value> extraYieldValues);
```
- **Location**: `Utils.cpp:314`
- **Purpose**: Copies original yield operands from `oldBlock`'s terminator, appends `extraYieldValues`, creates new `scf::YieldOp` at end of `newBlock`, and erases old yield.

### `replaceOpResultUses`
```cpp
void replaceOpResultUses(Operation *oldOp, Operation *newOp);
```
- **Location**: `Utils.cpp:333`
- **Purpose**: Replaces all uses of `oldOp` results with matching results from `newOp.getResults().take_front(numOldResults)`.

---

## 2. Block & Attribute Identification Helpers

### `getIfBlockId`
```cpp
static int getIfBlockId(scf::IfOp ifOp, int &outBlockId);
```
- **Location**: `UpdateConditionInfo.cpp:77`
- **Purpose**: Reads `IntegerAttr` with name `CVPipeline::kIf` (`"ssbuffer.if"`). Returns error if attribute is missing.

### `collectSSBufferIfOps`
```cpp
static void collectSSBufferIfOps(scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps);
```
- **Location**: `UpdateConditionInfo.cpp:88`
- **Purpose**: Walks `forOp` body and collects all `scf::IfOp` with attribute `kIf`.

### `countUniqueIfBlockIds`
```cpp
int countUniqueIfBlockIds(Operation *loopOp);
```
- **Location**: `Utils.cpp:215`
- **Purpose**: Walks operations inside `loopOp`, collecting unique `ssbuffer.if` block IDs.

### `isMainLoopOp`
```cpp
inline bool isMainLoopOp(Operation *op);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h:176`
- **Purpose**: Checks if an op is `scf::ForOp` or `scf::WhileOp` with attribute `CVPipeline::kMainLoop`.

### `hasFallbackAttr` / `setFallbackAttr`
```cpp
inline bool hasFallbackAttr(ModuleOp module);
inline void setFallbackAttr(ModuleOp module, int errorCode);
```
- **Location**: `third_party/ascend/include/DynamicCVPipeline/Common/Utils.h:80-95`
- **Purpose**: Checks or sets fallback flag on `ModuleOp` when dynamic CV pipeline compilation fails.

---

## 3. SSA Mapping & Constant Helpers

### `getLatestValue`
```cpp
static Value getLatestValue(const DenseMap<Value, Value> &map, Value key);
```
- **Location**: `UpdateConditionInfo.cpp:71`
- **Purpose**: Safely resolves the most recent SSA value for an iter_arg or control variable from `controlVarToLatestValue`. Returns `key` if no updated SSA value exists.

### `createIntConst`
```cpp
static Value createIntConst(OpBuilder &builder, Location loc, int64_t val, Type type);
```
- **Location**: `UpdateConditionInfo.cpp:76`
- **Purpose**: Helper to create `arith::ConstantIntOp` matching the bitwidth of `type` (defaults to 32 if not integer type). Extracts result `Value` directly.

---

## 4. Graph & Buffer Analysis Helpers

### `buildBufferDependencyMappings`
```cpp
static int buildBufferDependencyMappings(
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &buffers,
    DenseMap<Operation *, SmallVector<int>> &consumerToGroups,
    DenseMap<Operation *, SmallVector<int>> &outputToGroups);
```
- **Location**: `UpdateConditionInfo.cpp:189`
- **Purpose**: Generates reverse lookup tables (`consumerToGroups` and `outputToGroups`) from buffer dependency structures for $O(1)$ query during IR traversal.

### `buildOutputGroups`
```cpp
int buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar,
    SmallVector<OutputGroupInfo> &outputGroups);
```
- **Location**: `UpdateConditionInfo.cpp:365`
- **Purpose**: Deduplicates and groups producer operations and their associated control variables to generate capacity constraints (`var < group.outputs.size()`).

### `dfsTopologicalSort`
```cpp
static LogicalResult dfsTopologicalSort(
    Operation *op, DenseSet<Operation *> &visited, DenseSet<Operation *> &inStack,
    const DenseSet<Operation *> &ops, DenseMap<Operation *, int> *opOrder,
    SmallVectorImpl<Operation *> &sorted);
```
- **Location**: `Utils.cpp:89`
- **Purpose**: Topologically sorts operations within an `scf.for` body based on data dependencies, reporting cycles if present.

### `filterCrossCoreMapByForOp`
```cpp
static ConsumerProducerMap filterCrossCoreMapByForOp(scf::ForOp forOp, ConsumerProducerMap &crossCoreMap);
```
- **Location**: `UpdateLoopIterTimes.cpp:61`
- **Purpose**: Filters cross-core producer-consumer entries to only those where the consumer op resides within `forOp`.

### `getOtherScopeMainloop`
```cpp
static scf::ForOp getOtherScopeMainloop(ModuleOp module, bool currentIsCube, bool currentIsVector, int mainLoopId);
```
- **Location**: `UpdateLoopIterTimes.cpp:76`
- **Purpose**: Locates the counterpart main loop on the opposite core scope sharing the same `mainLoopId`.
