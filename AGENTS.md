# Development Guidelines & Troubleshooting (AGENTS.md)

## Toolchain & Build Paths
- **LLVM Environment Variable**:
  ```bash
  export PATH=~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/:$PATH
  ```
- **Build script**:
  ```bash
  bash ~/personal/build_ta.sh &> build/build.log
  ```
  *(Always redirect full build output to `build/build.log` to prevent context pollution).*
- **`triton-opt` path**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
  ```
- **`FileCheck` path**:
  ```bash
  ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck
  ```
- **Restoring in-tree sources**:
  After running build scripts that apply Ascend patches to Triton sources, clean the working tree via:
  ```bash
  python3 restore_sources.py
  ```

---

## MLIR Testing & FileCheck Pitfalls
- **Test execution pattern**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt <pass-flags> %s 2>&1 | ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck %s
  ```
- **Multiline op debug prints**:
  - Control-flow and scoped operations printed in debug logs (e.g., `Analyzing op scf.for ... { ... }` or multi-region ops) emit the entire nested body across multiple lines before printing trailing summaries or metadata. Avoid using `CHECK-NEXT:` immediately after matching such multiline ops; use `CHECK:` to find the target summary line or explicitly match the intervening lines.
- **Void-returning function calls**:
  - `void`-returning calls (e.g., `func.call @callee(...) : (...) -> ()`) are printed in IR and logs as `Analyzing op func.call @callee...` without an SSA result prefix (`%0 =`). Avoid writing `%{{.*}} = func.call` in `CHECK` patterns for void calls to prevent matching failures.

---

## General C++ & MLIR Implementation Best Practices
- **Preventing Dangling Pointers in Pipeline Passes**:
  - Data structures shared across passes (e.g., `ControlFlowConditionInfo *info` and `info->cntArgs`) hold raw `Operation *` keys.
  - When replacing operations in a pass (e.g., replacing `oldIfOp` with `newIfOp` and erasing `oldIfOp`), **always** erase or re-key entries in shared maps. Leaving erased pointers causes silent use-after-free crashes (e.g. `mlir::DictionaryAttr::contains` segfaults) in downstream passes.
- **`OpBuilder::create<OpType>` vs `Value`**:
  - In MLIR C++, `builder.create<OpType>(...)` returns the operation wrapper (`OpType`), not a `Value`. When passing the result as an operand or saving it in a `Value` map, always call `.getResult()` (or `.getResults()`).
- **`SmallVector` Inline Storage Size Constraints**:
  - LLVM's `SmallVector<T>` enforces `sizeof(T) <= 256` for default inline storage (`CalculateSmallVectorDefaultInlinedElements`). For struct types larger than 256 bytes, use `std::vector<T>` or `SmallVector<T, 0>`.
- **`DenseMap::lookup` Signature**:
  - LLVM `DenseMap::lookup(Key)` accepts a single parameter and returns default-constructed `Value` if missing. Never pass a default fallback value as a second argument; use an explicit `find()` helper instead.

---

## Guidelines for Subsequent Agents (经验与知识记录规范)
后续所有接手任务的 Agent **必须**严格遵守以下文档记录与维护规范：

1. **`AGENTS.md` 只保留最通用的准则**:
   - 仅记录全局通用的环境配置、编译命令、测试规范以及全局 C++/MLIR 编码陷阱。严禁将具体 Pass 的局部逻辑或细节直接堆积在根目录 `AGENTS.md` 中。
2. **Pass 专项文档放进 `AGENTS/<Topic>/<Pass>.md`**:
   - 针对具体模块或 Pass（如 `DynamicCVPipeline` 下的各 Pass），在 `AGENTS/<Topic>/` 下按 Pass 分解为独立的 `.md` 文件。
   - **必须同时记录旧版和新版的方案对比**：包括旧版为什么设计、存在什么缺陷，新版的核心机制、数据结构设计与实现原理。
   - 必须记录该 Pass 调试中踩过的坑（如悬垂指针、SSA 支配顺序等）及解决措施。
3. **`AGENTS/Utils.md` 记录可复用的辅助函数**:
   - 在阅读或重构代码时，只要发现或实现了具备复用价值的辅助函数（无论是定义在 `.h` 中还是隐藏在 `.cpp` 中的内部 helper），**必须**额外记录到 `AGENTS/Utils.md` 中。
   - 记录要素包括：完整函数签名、所属文件及大致行号、功能用途说明，以便后续需要时随时外提为通用公共组件。

---

## Document Index
- **Dynamic CV Pipeline (`AGENTS/DynamicCVPipeline/`)**:
  - [Overview](AGENTS/DynamicCVPipeline/Overview.md): 全局管线流程与两核解耦机制
  - [CloneOps (Step 0)](AGENTS/DynamicCVPipeline/CloneOps.md): 消除跨块算子共享
  - [ProcessArgs (Step 1)](AGENTS/DynamicCVPipeline/ProcessArgs.md): 消除跨块循环参数共享
  - [CreateIfOps (Step 2)](AGENTS/DynamicCVPipeline/CreateIfOps.md): 封装初始 block If 分支
  - [InitDependentMap (Step 3)](AGENTS/DynamicCVPipeline/InitDependentMap.md): 提取核间/核内核间依赖与 DAG
  - [UpdateLoopOps (Step 4)](AGENTS/DynamicCVPipeline/UpdateLoopOps.md): 循环扩展初始计数器与状态
  - [UpdateConditionInfo (Step 5)](AGENTS/DynamicCVPipeline/UpdateConditionInfo.md): **核心重构：旧版 SSBuffer 方案 vs 新版纯标量锁步仿真方案**
  - [UpdateLoopIterTimes (Step 6)](AGENTS/DynamicCVPipeline/UpdateLoopIterTimes.md): 循环迭代扩展与归纳变量替换
- **Reusable Utilities & Functions**:
  - [Utils Index](AGENTS/Utils.md): 跨 Pass 辅助工具函数库（循环构建迁移、属性提取、SSA 辅助等）
