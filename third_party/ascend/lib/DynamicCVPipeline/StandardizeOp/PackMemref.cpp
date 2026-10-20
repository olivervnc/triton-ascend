#include <memory>

#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

using namespace mlir;
using namespace CVPipeline;

static llvm::LogicalResult pack(memref::AllocOp allocOp) {}

namespace {

class PackMemrefPass
    : public PassWrapper<PackMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackMemrefPass);

  PackMemrefPass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<scope::ScopeDialect, hivm::HIVMDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pack-memref";
  }
};

} // namespace

void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  SmallVector<memref::AllocOp> allocs;
  moduleOp.walk([&allocs](memref::AllocOp alloc) { allocs.push_back(alloc); });
  for (auto allocOp : allocs) {
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CVPipeline::createPackMemrefPass() {
  return std::make_unique<PackMemrefPass>();
}
