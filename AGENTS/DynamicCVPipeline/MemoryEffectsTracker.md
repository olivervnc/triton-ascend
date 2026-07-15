# MemoryEffectsTracker

跨 pass 共享的内存依赖图（`include/DynamicCVPipeline/Common/MemoryEffectsTracker.h` + `lib/DynamicCVPipeline/Common/MemoryEffectsTracker.cpp`）。`analyzeOp` 按程序序扫块，`collectOuterEffects` 把每个算子的 memref 效果归并到 `MemSlot`，`applyEffects` 把效果写回槽状态，`collectPreds` / `recordEdges` 据此产出四个查询：`getMemUsers`（来自 `defs`）、`getExecBefore` / `getExecAfter`（来自 `preds`）、`getRealDependency`（同样来自 `preds`）。

消费者：`DataDependencyAnalysis`、`RefineArgsBlockId`、`FlagIdReuse`、`SplitIfByBlockId`、`UBUsageOptPass`、`CloneOps`、`DependencyHelper`、`Common/Utils.cpp`（`getUpstreamOpsWithMemoryDeps`，供 OpClassifier 的 CUBE/VECTOR 传播）。

## 陷阱

- **`MemSlot` 依赖边不可裁剪**: `applyEffects` 对每个 memref 结果槽无条件写 `s->lastWriter = op;`（`for (auto result : op->getOpResults())` 块内那处；另有 unknown / allocate / write 三处）。这条 RAW pred 经 `collectPreds` 同时喂给 `recordEdges`（`execBefore` / `execAfter` 边）与 `getRealDependency`（跨核 set/wait），看起来像"产出不算写"的合理精简，实测劣化性能（daebd90 顺手删过该行，是回滚原因之一）-> 改 `MemSlot` / `applyEffects` 前先确认 4 处 `lastWriter = op;` 仍在（`rg -n 'lastWriter = op;' .../Common/MemoryEffectsTracker.cpp`）。
