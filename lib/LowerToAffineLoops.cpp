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

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

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

  struct ConstantOpLowering : public OpRewritePattern<flow::ConstantOp> {
    using OpRewritePattern<flow::ConstantOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(flow::ConstantOp op, PatternRewriter &rewriter) const final {
      llvm::errs() << "ConstantOpLowering\n";

      DenseElementsAttr constantValue = op.getValue();
      Location loc = op.getLoc();

      auto tensorType = op.getType().cast<TensorType>();
      auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

      auto valueShape = memRefType.getShape();
      SmallVector<Value, 8> constantIndices;
      if (!valueShape.empty()) {
        for (auto i: llvm::seq<int64_t>(0, *std::max_element(valueShape.begin(), valueShape.end()))) {
          constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
        }
      } else {
        constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      SmallVector<Value, 2> indices;
      auto valueIt = constantValue.value_begin<FloatAttr>();
      llvm::errs() << valueShape.size() << "\n";

      std::function<void(uint64_t)> storeElements = [&](uint64_t dim) {
        if (dim == valueShape.size()) {
          rewriter.create<AffineStoreOp>(
                  loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++),
                  alloc,
                  llvm::ArrayRef(indices));
          return;
        }

        for (uint64_t i = 0, e = valueShape[dim]; i != e; ++i) {
          indices.push_back(constantIndices[i]);
          storeElements(dim + 1);
          indices.pop_back();
        }
      };
      storeElements(0);

      rewriter.replaceOp(op, alloc);

      return success();
    }
  };

  struct PrintOpLowering : public OpConversionPattern<flow::PrintOp> {
    using OpConversionPattern<flow::PrintOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(flow::PrintOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
      rewriter.updateRootInPlace(op, [&] {
        op->setOperands(adaptor.getOperands());
      });

      return success();
    }
  };

}// namespace

namespace {
  struct FlowToAffineLowingPass : public PassWrapper<FlowToAffineLowingPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlowToAffineLowingPass)
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
    }
    void runOnOperation() final;
  };
}// namespace

void FlowToAffineLowingPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
  target.addIllegalDialect<flow::FlowDialect>();
  // 确保TensorType已经全部转换为MemRef
  target.addDynamicallyLegalOp<flow::PrintOp>([](flow::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering, ReturnOpLowering, ConstantOpLowering, PrintOpLowering>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::flow::createLowerToAffinePass() {
  return std::make_unique<FlowToAffineLowingPass>();
}
