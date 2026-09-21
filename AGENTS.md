# Development Guidelines & Troubleshooting (AGENTS.md)

## Toolchain & Build Paths
- **Build script**:
  ```bash
  bash ~/personal/build_ta.sh
  ```
  *(Note: Compilation output can be very long; avoid dumping or reading full logs unnecessarily).*
- **`triton-opt` path**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
  ```
- **`FileCheck` path**:
  ```bash
  ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck
  ```

## MLIR Testing & FileCheck Pitfalls
- **Test execution pattern**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt <pass-flags> %s 2>&1 | ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck %s
  ```
- **Multiline op debug prints**:
  - Control-flow and scoped operations printed in debug logs (e.g., `Analyzing op scf.for ... { ... }` or multi-region ops) emit the entire nested body across multiple lines before printing trailing summaries or metadata. Avoid using `CHECK-NEXT:` immediately after matching such multiline ops; use `CHECK:` to find the target summary line or explicitly match the intervening lines.
- **Void-returning function calls**:
  - `void`-returning calls (e.g., `func.call @callee(...) : (...) -> ()`) are printed in IR and logs as `Analyzing op func.call @callee...` without an SSA result prefix (`%0 =`). Avoid writing `%{{.*}} = func.call` in `CHECK` patterns for void calls to prevent matching failures.
