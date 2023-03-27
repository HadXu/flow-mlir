//
// Created by lei on 2023/3/27.
//
#include "FlowDialect.h"
#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;


namespace {
  struct FuncOpLowering : public OpConversionPattern<flow::FuncOp> {
    using OpConversionPattern<flow::FuncOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(flow::FuncOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {

      llvm::errs() << "FuncOpLowering\n";
      if (op.getName() != "main") {
        return failure();
      }

      if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
        return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
          diag << "expected 'main' to have 0 inputs and 0 results";
        });
      }

      auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
      rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

  struct ReturnOpLowering : public OpRewritePattern<flow::ReturnOp> {
    using OpRewritePattern<flow::ReturnOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(flow::ReturnOp op,
                                  PatternRewriter &rewriter) const final {
      llvm::errs() << "ReturnOpLowering\n";
      if (op.hasOperand())
        return failure();
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
      return success();
    }
  };
}// namespace

namespace {
  struct FlowToAffineLowingPass : public PassWrapper<FlowToAffineLowingPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlowToAffineLowingPass)
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<func::FuncDialect>();
    }
    void runOnOperation() final;
  };
}// namespace

void FlowToAffineLowingPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
  target.addIllegalDialect<flow::FlowDialect>();
  target.addDynamicallyLegalOp<flow::PrintOp>([](flow::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering, ReturnOpLowering>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::flow::createLowerToAffinePass() {
  return std::make_unique<FlowToAffineLowingPass>();
}
