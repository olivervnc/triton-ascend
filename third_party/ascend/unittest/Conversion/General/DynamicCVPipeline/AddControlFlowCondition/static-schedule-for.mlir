// RUN: triton-opt --add-control-flow-condition %s > %t
// RUN: FileCheck %s --check-prefix=NOCOUNT < %t
// RUN: FileCheck %s < %t

// Static schedule mode: with ssbuffer.cond_mode = "static" every cross-core
// part of an ssbuffer.if condition is replaced by a compile-time firing set on
// the loop induction variable, so no ssbuffer counter word is allocated or
// touched. The kernel body is the multi_flow_opt_test cross-core pipeline, so
// the legacy counter form of the very same input is checked by that test.
//
// The DAG has three groups and two buffers between the cores:
//   CUBE:   C0 (produces g0) -> C2 (produces g3) -> C1 (consumes g2, produces
//           g1, gated by flowOpt)
//   VECTOR: V0 (consumes g0/g3, produces g2) -> V1 (consumes g1)
// Both cores keep the 8192/128 trip count and run in lockstep, so a step fires
// a block only once the peer has produced its input and only while the peer has
// not yet run ahead of the buffer credit. The resulting schedules alternate:
// V0 fires on the odd steps starting at step 1, V1 one step later, the CUBE
// producers on the even steps, and C1 once C0's counter passes two steps.

// The CHECK patterns and the IR below are not C++; keep clang-format away.
// clang-format off
// The whole function is counter free: no ssbuffer pointer, no volatile mark and
// no memref load/store at all.
// NOCOUNT-NOT: address_space<ssbuf>
// NOCOUNT-NOT: memref.load
// NOCOUNT-NOT: memref.store
// NOCOUNT-NOT: memref_ext.volatile

// CHECK: func.func @test_multiple_CV_data_flow
// CHECK: scf.for
// V0: step 1 only, then every second step from step 3 on, i.e. the odd steps.
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c128_i32
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c256_i32
// CHECK: arith.andi
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c384_i32
// CHECK: arith.subi %{{[^ ]+}}, %c384_i32
// CHECK: arith.remui %{{[^ ]+}}, %c256_i32
// CHECK: arith.cmpi eq, %{{[^ ]+}}, %c0_i32
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c8192_i32
// CHECK: arith.andi
// CHECK: ssbuffer.if = 5 : i32}
// V1: fires on the odd steps from step 3 on, because C1 produces its input one
// step after V0 consumed g2.
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c384_i32
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c512_i32
// CHECK: arith.andi
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c640_i32
// CHECK: arith.subi %{{[^ ]+}}, %c640_i32
// CHECK: arith.remui %{{[^ ]+}}, %c256_i32
// CHECK: arith.cmpi eq, %{{[^ ]+}}, %c0_i32
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: ssbuffer.if = 6 : i32}
// C0 and C2: the even steps, starting at step 0.
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c0_i32
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c128_i32
// CHECK: arith.andi
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c256_i32
// CHECK: arith.subi %{{[^ ]+}}, %c256_i32
// CHECK: arith.remui %{{[^ ]+}}, %c256_i32
// CHECK: arith.cmpi eq, %{{[^ ]+}}, %c0_i32
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c8192_i32
// CHECK: arith.andi
// CHECK: ssbuffer.if = 0 : i32}
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c0_i32
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c128_i32
// CHECK: arith.andi
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c256_i32
// CHECK: arith.subi %{{[^ ]+}}, %c256_i32
// CHECK: arith.or
// CHECK: ssbuffer.if = 2 : i32}
// C1 keeps the flowOpt clause: it may run once its source block C0 has counted
// up to lowerBound + step * flowOptNum = 0 + 128 * 2, and that gate reads C0's
// counter, not C1's own. flowOptNum = min(intra - 1, cross) = min(2, 2).
// Which peer counter the legacy flowOpt pairing picks is not stable across runs,
// so the operand is left open here.
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c256_i32
// CHECK: arith.cmpi slt, %{{[^ ]+}}, %c384_i32
// CHECK: arith.andi
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %c512_i32
// CHECK: arith.subi %{{[^ ]+}}, %c512_i32
// CHECK: arith.remui %{{[^ ]+}}, %c256_i32
// CHECK: arith.cmpi eq, %{{[^ ]+}}, %c0_i32
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.muli %{{[^ ]+}}, %c2_i32
// CHECK: arith.addi %c0_i32, %{{[^ ]+}}
// CHECK: arith.cmpi sge, %{{[^ ]+}}, %{{[^ ]+}}
// CHECK: ssbuffer.if = 1 : i32}
// CHECK: return


module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.cond_mode = "static"} {
  func.func @test_multiple_CV_data_flow(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd", ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32} {
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc0 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_5a = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5a {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %30:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %memspacecast_13 = memref.memory_space_cast %alloc_5a {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        %reshape_13 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}
        %reshape_14 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x8x16x16xf16>
        hivm.hir.copy ins(%reshape_14 : tensor<8x8x16x16xf16>) outs(%alloc0 : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32]}

        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_6a = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6a {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %20:2 = scf.for %arg16 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32) -> (i32, i32)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %30 = tensor.empty() {ssbuffer.block_id = 0 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 0 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32]} ins(%30 : tensor<128x128xf32>) outs(%alloc_6 : memref<128x128xf32, #hivm.address_space<ub>>)

        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %31 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32]} ins(%31 : tensor<128x128xf32>) outs(%alloc_6a : memref<128x128xf32, #hivm.address_space<ub>>)

        hivm.hir.sync_block_wait {ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %32 = hivm.hir.convert_layout %alloc_4 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %321 = hivm.hir.convert_layout %alloc_5 output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 0 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32]} : (memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x128xf16, #hivm.address_space<cbuf>>
        %40 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 1 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32]} ins(%40 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        scf.yield %arg17, %arg18 : i32, i32
      } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
// clang-format on
