# Triton-Ascend Development Guidelines & Invariants (AGENTS.md)

## 1. Toolchain, Build & Test Entry Points
- **LLVM PATH**: `export PATH=~/.triton/llvm/llvm-ubuntu-x64/bin/:$PATH` (stable symlink; the versioned `llvm-f6ded0be-*-ubuntu-x64` directory changes with the LLVM pin).
- **Full build (mandatory on a fresh or unconfigured tree)**:
  ```bash
  bash ~/personal/build_ta.sh &> build/build.log
  ```
  Never dump raw build output into context; redirect to `build/*.log` (never `/tmp`, which is tmpfs) and read the failing lines only.
- **Incremental build (after `build_ta.sh` has configured the tree)**:
  ```bash
  nice -n 5 ninja -j6 -C build/cmake.linux-x86_64-cpython-3.11 triton-opt &> build/ninja.log
  ```
- **Binaries & test runners**:
  - `triton-opt`: `build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt`
  - `FileCheck`: `~/.triton/llvm/llvm-ubuntu-x64/bin/FileCheck`
  - **Single UT**: `build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt <pass-flags> <test.mlir> 2>&1 | ~/.triton/llvm/llvm-ubuntu-x64/bin/FileCheck <test.mlir>`
  - **Whole DCV UT suite**: `python3 run_dcv_ut.py` (globs `<dir>/**/*.mlir` recursively, so subdir suites such as `ComputeBlockOpt/SplitIfByBlockId/` are included; `lit`/`llvm-lit` are unavailable).
  - **Clean patched in-tree sources**: `python3 restore_sources.py`
  - **Local scripts index**: `AGENTS/LocalTooling.md` (placement rules, environment facts).
- **Project skills**: `.agents/skills/<name>/SKILL.md` are discovered from the working directory up to the repository root (Agent Skills standard, no `settings.json` entry needed). They hold executable procedures only; invariants stay in this file and `AGENTS/<Topic>/`. `.agents/**` is listed in the `private` fileset, so a skill never leaks into the pushed code track unless you move it out deliberately.

### Build & Repo Constraints
- **Bound every build to `-j6` (max `-j8`)**: a bare `ninja` spawns `nproc+2` (26) jobs, each clang-22 TU peaking at 1-2.2 GiB RSS, and OOM-freezes the host. Same for `cmake --build`.
- **Never build on an unconfigured or stale tree**: the in-tree patch `third_party/ascend/patch/triton-ascend-3.6.0.patch` injects `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__` into the top-level `CMakeLists.txt` and overlays `bin/`/`python/`/`lib/Dialect`. Without it the build fails with `no member named 'CopyOpInterface' in namespace 'mlir'` even when the cache still reports `LLVM_MAJOR_VERSION_22_COMPATIBLE:BOOL=ON`. Always configure through `bash ~/personal/build_ta.sh`; never hand-edit `CMAKE_CXX_FLAGS`.
  - **Same error, different cause**: a revision that simply does not carry the patch files (a bare `jj new` on the feature commit, or a freshly split commit) fails identically — build on a child revision that restored those paths instead of chasing the stale-tree explanation.
  - **Recovery after `restore_sources.py`**: `git apply third_party/ascend/patch/triton-ascend-3.6.0.patch && nice ninja -j6 -C build/<cfg> triton-opt` is enough (the `CMakeLists.txt` change makes cmake re-run); run `python3 restore_sources.py` once the build finishes.
- **`build_ta.sh` leaves 17 tracked files dirty** (`CMakeLists.txt`, `bin/`, `python/`, `lib/Dialect`, `include/triton`) -> commit explicit paths only, never `git add -A`, and run `python3 restore_sources.py` **before** `request_code_review` (the gate diffs the real `@`). Never judge `@`'s cleanliness through a display filter: a `jj diff --summary | rg -v 'CMakeLists|bin/|...'` hides exactly the paths that break the commit.
- **Proving UT coverage and equivalence**:
  - **A new branch needs only the red half**: back up the file, guard the new code with `if (false && ...)`, rebuild (1 TU + link), confirm the UT fails, then restore. This proves the UT covers the fix instead of passing on the old tree too.
  - **A behaviour-neutral step** is proven by the UT green on both the old and the new tree and red on the buggy intermediate.
  - **A structural refactor** is best proven by byte-identical pass output: dump per-UT output to `build/<step>_baseline/` before and after, then `diff -rq` — but prove determinism first (see below), otherwise prefer structural/count invariants.
  - **Rebuilding an old revision**: the in-tree patch lives only in the working-copy commit, so carry it over (`jj new <old-rev>` + `jj restore --from <wc-commit> $(jj diff --name-only -r <wc-commit>)`).
- **The DCV pipeline is not deterministic**: `AllocMultiCache/AddMultiBufferInnerScope.cpp` iterates a `DenseMap<Value, ...>`, so buffer pairing differs run to run (`mla_decode_split_in` produced three different md5s from one binary). Hash two runs before believing any byte comparison.
- **Debugging stripped binaries (`safe_compile.cmake`)**: prefer `triton-opt <flags> -debug` / `-debug-only=<type>` over `gdb`. `safe_compile.cmake` strips at link time (`-s`) while `.o` files keep `-g`; drop `-s` from the link flags and relink (~2s, no recompile) for a crash backtrace. Never commit edits to `safe_compile.cmake`.
- **LSP diagnostics are advisory**: CMake mirrors the tree into `build/cmake.*/`, so clangd can resolve a stale copy after a header move and report phantom errors. The `ninja` exit status is the arbiter; never edit code just to silence a squiggle.
- **Vendored sub-repos**: `third_party/ascend/AscendNPU-IR` is an independent git repo; never include its changes in a commit here.
- **Dual-track (dual-tentacle) delivery**: every PR runs a pure-code track `<PR>` on `armbase()` (the only thing pushed, never containing a `private` file) plus a binary doc ladder `hack-<PR>` whose k-th node is `merge(H_{k-1}, P_k)`. Drive it through `jj mnew`/`mput`/`mcommit`/`mgo`/`mback`/`mland`; the invariants, the three knobs and the conflict procedure live in [Jj Octopus Workflow](AGENTS/JjOctopus.md).

---

## 2. MLIR, FileCheck & Cross-Pass Pitfalls

### FileCheck & IR Inspection
- **Multiline op debug prints**: scoped/multi-region ops (`scf.for`, `scf.if`) print their whole body before trailing debug lines -> use `CHECK:`, never `CHECK-NEXT:` right after the op header.
- **Void calls** emit no SSA prefix: never match `%{{.*}} = func.call` for `-> ()`.
- **`CHECK-LABEL` cannot define a variable**: split into `// CHECK-LABEL: func.func @f(` + `// CHECK-SAME: %[[A:[a-zA-Z0-9_.]+]]:`; a `%[[A:%.*]]` capture is greedy and swallows the rest of the argument list.
- **`bufferization.to_tensor` printing**: the pinned LLVM omits ` to tensor<...>` (`... : memref<...>`) while the parser still accepts it in input IR -> omit the suffix in `CHECK` lines, keep it in the UT input.
- **Predicates print with a comma**: `arith.cmpi sge, %x, %y`; a regex written as `sge\s+%x` matches nothing.
- **Reading printed IR with a script**:
  - **Region nesting**: track an indent stack of `(indent, opener)`; never count `{}` (attributes use them too, and one region is not one nesting level). Match **both** print forms: `%0 = scope.scope : () -> ... {` and the generic `%73:17 = "scope.scope"() ({`.
  - **`scope.scope` attributes print on the closing `}` line** (`ssbuffer.block_id`, `ssbuffer.core_type`, `ssbuffer.load`), so a unit's attributes are on `seg[end]`, not `seg[end-1]`.
  - **SSA names are reused across regions** (`%alloc_50` can appear in several `scf.if`/`scf.for` bodies): build per-region symbol tables, and resolve `memref.subview`/`memref.cast` aliases to the root alloc before claiming "this alloc has no writer" (`python3 find_unwritten_allocs.py`).
  - **Moved versus cloned**: a cloned op keeps the original SSA names and attributes, while a moved op no longer exists outside the scope -> strip only the `%res = ` prefix and compare the whole line; normalising SSA names to `%v` turns distinct same-shape ops into false "clones".
  - **Did the pass run at all?** Read the `IR Dump After <Pass> (<arg>)` headers from `--mlir-print-ir-after-all`. Registration source, bookmarks, or the ledger are not evidence: the checkout and the binary may be out of sync.

### IR Traversal, Dominance & Mutation
- **`WalkResult::skip()` requires `PreOrder`**: `op.walk` defaults to `PostOrder`, which visits regions first and ignores `skip()` -> spell `op.walk<WalkOrder::PreOrder>(...)` when pruning a subtree (e.g. `scope::ScopeOp`).
- **`scope::ScopeOp` is a `RegionBranchOpInterface`**: helpers that derive a core type by walking regions see an unclassified body if the scope was skipped -> read the `kCoreType` attribute directly, or walk to the nearest enclosing `scope::ScopeOp`.
- **No IR mutation inside `op->walk`**: collect targets into a `SmallVector<Operation *>` first; when replacing or erasing an op, update shared raw-pointer maps (e.g. `ControlFlowConditionInfo::cntArgs`) in the same step.
- **Never iterate a container a helper may append to**: snapshot the candidates before calling e.g. `splitOperationForCubeAndVector` (it pushes into `allOps`).
- **Container iteration order is observable**: swapping `DenseMap`->`MapVector` on a shared structure (`BlockDependencyGraph::blockNodes`) changes the greedy merge selection in `MergeCubeBlock` -> treat it as a behaviour change and re-baseline the UTs.
- **Dominance**:
  - Value/op dominance: use `mlir::DominanceInfo::properlyDominates`; never hand-roll block ancestry (it treats every `BlockArgument` as dominating and ignores in-block order).
  - Ordering a batch of ops during mutation: `DominanceInfo` cannot be used (it would be stale after every clone/move, and it asserts on single-block regions) -> use the stateless `CVPipeline::isBeforeInIr` / `sortByDomOrder` (`DynamicCVPipeline/Common/IROrder.h`), which does the two-pointer LCA and `findAncestorOpInBlock` and reports sibling regions as unordered.
  - Never call `isBeforeInBlock` across blocks (assert/UB).
- **Avoid redundant `op->walk`**: `val.getUsers()`, direct `block` iteration and `block->findAncestorOpInBlock(*op)` answer local questions.
- **Closure ops may come from a dominating ancestor block**: mask-apply zero fills and `tensor.empty` inits are captured from an outer block by design; a same-block guard silently drops them. Only the arith/linalg *chain* follows its own same-block rule.
- **`PatternRewriter` discipline**: inside `RewritePattern`/`ConversionPattern` route every mutation through `rewriter.*`, and wrap temporary insertion-point changes in `OpBuilder::InsertionGuard guard(builder);`.
- **A new UT case must not reuse an existing `block_id`**: `willCreateCycle` builds its group from `bm.getOpsByBlockId(id)` across the whole module, so a reused id joins another function's group and gets that group's unification rejected as cyclic. A second top-level `module` does not isolate it.

### Pass Architecture, C++ API & Diagnostics
- **Pass thread-safety**: never store per-invocation mutable state (`DenseMap`, worklists, `IRRewriter`) as a `Pass` member; hold it in a stack-allocated helper inside `runOnOperation()`. Add methods to the `Pass` only for `protected` APIs.
- **Analysis invalidation**: mutating CFG/block connectivity invalidates `DominanceInfo`/`PostDominanceInfo` -> re-query or `getAnalysisManager().invalidate(...)`, never mark them preserved across CFG edits.
- **Dialect registration**: override `getDependentDialects(DialectRegistry &)` whenever the pass creates ops from dialects the input IR does not guarantee.
- **Attribute constants**: never hardcode `"ssbuffer.*"` or core-type strings; use `DynamicCVPipeline/Common/Utils.h` and `ScopeOpUtils.h` (`kCoreType`, `kBlockId`, `kScopeLoadAttr`, `kCoreTypeCube`, `kCoreTypeVector`, …).
- **`namespace triton` arrives only via `ComputeBlockOpt/Passes.h`**: a TU split off a `lib/DynamicCVPipeline/.../<Pass>.cpp` that keeps `using namespace triton;` must include it, else `error: expected namespace name`.
- **API gotchas**:
  - `builder.create<OpType>(...)` returns the op, not a `Value`; call `.getResult()`/`.getResults()`.
  - `DenseMap`/`SmallDenseMap` have no `lookupOrDefault`: `lookup(k)` default-constructs on miss, `lookup_or(k, default)` takes a custom default, `find(k)` distinguishes a miss; prefer `.contains()` over `.count()`.
  - `SmallVector<T, 0>` only where the container substitutes `std::vector` (bulk storage), with a comment saying why; default `SmallVector<T>` for locals, params and returns.
  - `StringRef` does not own memory: never bind it to a temporary `.str()` or `Twine`; pass `ArrayRef`/`ValueRange`/`StringRef` instead of copying containers.
  - Never silence `[[nodiscard]] LogicalResult` with `(void)fn(...)`; log the skip reason with `LLVM_DEBUG`.
  - `scf::IfOp`'s empty-`TypeRange` builder already yields a result-less `scf.if`; but the implicit `scf.yield` is a parser convenience, so after erasing it rebuild the terminator unconditionally (an `if (hasElse)`-guarded rebuild leaves `block with no terminator`).
  - Diagnostics: disambiguate `Op` wrappers with `<< op.getOperation()` or `<< *op`.
  - Assume valid IR in safety predicates: record structural guarantees as `Preconditions:` in the doc comment, and prefer `llvm::zip` over `zip_equal`.
- **Collapsing two disjoint lists into one** silently double-counts every consumer that iterated both -> delete the secondary accessor in the same change and pin the shape in a UT.
- **Logging**: `LLVM_DEBUG` + `llvm::dbgs()` via the TU's variadic `LDBG(...)`; one lowercase line, no trailing punctuation. `LLVM_DEBUG` is a runtime `DebugFlag` test, so keep string construction inside the macro.
- **A real failure is a diagnostic, not a log** (`emitError`/`emitWarning`/`emitRemark`), except under `AddDynamicCVPipelinePass`: its `ScopedDiagnosticHandler` swallows diagnostics in release builds, so trace decisions and skip reasons through the pass's `DEBUG_TYPE` and keep `--debug-only=<type>` sufficient to read the whole chain.
- **DCV failure protocol**: a pass that can continue with degraded input ends with `setFallbackAttr(module, ERRCODE_FAILED|ERRCODE_IGNORED) + return` so downstream passes short-circuit; replacing that pair with `signalPassFailure()` loses both the short-circuit and the recorded reason.
- **No core dump in DCV**: a suspected-impossible state returns `failure()` or the fallback attr plus an `LDBG` reason — never `llvm_unreachable`/`assert`, including inside debug-only helpers (an enum-to-string switch ends with `return "Unknown";`).

### Code Style & Formatting
- **UT `.mlir` files are never clang-formatted**: FileCheck patterns exceed `ColumnLimit: 80` by design. `third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/.clang-format` sets `DisableFormat: true`; a new UT tree outside such a directory fails any clang-format precheck.
- **Local headers use the full prefix**: `#include "ascend/include/DynamicCVPipeline/..."`. A bare `#include "DynamicCVPipeline/..."` compiles (both roots are on the include path) and gets rejected in review. Changing a line to the full form is not a text substitution: `.clang-format` puts `^"ascend/` in priority 5 and the rest in 6, so re-run `clang-format -i` on the file afterwards (`--dry-run --Werror` fails otherwise).
- **Two clang-format versions exist**: `.pre-commit-config.yaml` pins `mirrors-clang-format v19.1.6`, the review gate runs the `PATH` binary (currently 23.x). Both can disagree on the same file, and the pre-commit hook reformats whole files regardless of what the review gate checked.
- **`InsertBraces: true` makes a whole-file `clang-format -i` churn**: one legacy brace-less `if`/`for` adds dozens of unrelated lines (`MergeComputeBlockPass.cpp` had 20+). Either scope the format check to the changed lines or declare that churn explicitly in the same diff; do not skip the gate.
- **UT-first / comment / doc conventions**: English comments only; `struct` only when all members are public (`public:` -> `protected:` -> `private:`, private helpers before data); `///` Doxygen on public APIs in `.h` and private APIs in `.cpp` (never duplicated), directly above each enum value; 80-column `//===---...---===//` banners only for macro file sections.
- **Idioms**: prefer pure `static` functions returning `LogicalResult`/`FailureOr<T>` over out-parameters; `llvm::any_of`/`llvm::all_of` over hand-rolled boolean loops; pass composite handles (`OpResult`, `OpOperand`) instead of `(Operation *, size_t)`.
- **Change scope**: fix only what the change touches and report pre-existing style issues instead of reformatting them. Exception: the review gate runs `clang-format --dry-run --Werror` over the whole changed file, so one pre-existing violation blocks it — run `clang-format -i` and declare the extra churn.

---

## 3. Knowledge Maintenance Rules (Mandatory)
1. **Keep root `AGENTS.md` general**: repo-wide build/test commands and cross-pass MLIR/C++ invariants only. Pass-specific logic belongs in `AGENTS/<Topic>/<Pass>.md`; a **shared module's** invariants (`MemoryEffectsTracker`, `ScalarClosure`, `ScopeOpUtils`) live in its own `AGENTS/<Topic>/<Module>.md` linked from §4. Unattachable one-line traps stay in §2.
2. **Pass docs (`AGENTS/<Topic>/<Pass>.md`)** carry the **Old vs. New design comparison**, pass-specific pitfalls as an invariant + `Rejected: <approach> -> <why it fails>` where a tempting design was proven unviable. Design depth is welcome there; the root file stays an index.
3. **Entry format**: one rule per bullet, 1-3 lines, <= 600 chars, timeless present tense (`- **[Trap]**: [root cause] -> [exact rule]`). Never stitch several `**bold**` topics into one paragraph, never write first-person session narrative (`real case:`, `我据此`), benchmark or line counts, temporary SSA names, or a `Notes (...)` append — merge into the existing bullet or delete it.
4. **Reusable helper index (`AGENTS/Utils.md`)**: C++ helpers only (`signature`, `file:line`, purpose, consumers), never the rationale or pitfalls (those link to a design doc). CLI scripts belong in `AGENTS/LocalTooling.md`.
5. **Helper header placement**: keep `DynamicCVPipeline/Common/Utils.h` small. A new self-contained helper group gets its own topic header under `include/DynamicCVPipeline/<Topic>/` (deps limited to what it needs) plus the matching `lib/DynamicCVPipeline/<Topic>/`, indexed in `AGENTS/Utils.md`. Both `lib/.../<Pass>/`-relative and `ascend/include/...` spellings resolve; shared headers never live under a pass's private subdirectory.

---

## 4. Document Index
- **Dynamic CV Pipeline (`AGENTS/DynamicCVPipeline/`)**:
  - [Overview](AGENTS/DynamicCVPipeline/Overview.md): 管线流程、两核解耦、`block_id` 是调度分组
  - [CloneOps (Step 0)](AGENTS/DynamicCVPipeline/CloneOps.md): 消除跨块算子共享
  - [ProcessArgs (Step 1)](AGENTS/DynamicCVPipeline/ProcessArgs.md): 消除跨块循环参数共享
  - [CreateIfOps (Step 2)](AGENTS/DynamicCVPipeline/CreateIfOps.md): 封装初始 block If 分支
  - [InitDependentMap (Step 3)](AGENTS/DynamicCVPipeline/InitDependentMap.md): 核间/核内核间依赖与 DAG
  - [UpdateLoopOps (Step 4)](AGENTS/DynamicCVPipeline/UpdateLoopOps.md): 循环扩展初始计数器与状态
  - [UpdateConditionInfo (Step 5)](AGENTS/DynamicCVPipeline/UpdateConditionInfo.md): 旧版 SSBuffer 方案 vs 新版纯标量锁步仿真
  - [UpdateLoopIterTimes (Step 6)](AGENTS/DynamicCVPipeline/UpdateLoopIterTimes.md): 迭代扩展与归纳变量替换
  - [CustomOpAndScope](AGENTS/DynamicCVPipeline/CustomOpAndScope.md): CustomOp 识别、scope 临时封装与无损解包、mask 链打包
  - [ScalarClosure](AGENTS/DynamicCVPipeline/ScalarClosure.md): 标量/视图闭包收集与物化
  - [Closure (未来计划)](AGENTS/DynamicCVPipeline/Closure.md): 统一闭包收集（未实施）
  - [SplitIfByBlockId](AGENTS/DynamicCVPipeline/SplitIfByBlockId.md): `scf.if` 按 `block_id` 拆分与 `createPlaceholderValue` 占位
  - [SinkI1ProducersIntoUsers](AGENTS/DynamicCVPipeline/SinkI1ProducersIntoUsers.md): i1 生产者下沉与跨 core 的 `block_id` 组
  - [DataDependencyAnalysis](AGENTS/DynamicCVPipeline/DataDependencyAnalysis.md): `BlockInfo` 分组、`isCube` 取 `ops[0]`、V2C/C2V/C2C 记录条件
  - [MemoryEffectsTracker](AGENTS/DynamicCVPipeline/MemoryEffectsTracker.md): 跨 pass 内存依赖图（`MemSlot` / `getExecBefore` / `getRealDependency`）
- **Local Tooling & Workflow**:
  - [Local Tooling](AGENTS/LocalTooling.md): 本地脚本索引、放置规则、构建环境事实
  - [Jj Octopus Workflow](AGENTS/JjOctopus.md): 双轨制 jj 阶梯工作流（`mnew`/`mput`/`mcommit`/`mgo`/`mback`/`mland`）
- **Reusable Utilities**:
  - [Utils Index](AGENTS/Utils.md): 跨 Pass 辅助工具函数库索引
