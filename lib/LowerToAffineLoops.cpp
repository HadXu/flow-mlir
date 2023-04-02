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

using LoopIterationFn = function_ref<Value(OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {

  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  auto memRefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);// 2
  SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

  buildAffineLoopNest(
          rewriter, loc, lowerBounds, tensorType.getShape(), steps, [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
          });
  rewriter.replaceOp(op, alloc);
}


namespace {
  template<typename BinaryOp, typename LoweredBinaryOp>
  struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx)
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
      auto loc = op->getLoc();
      lowerOpToLoops(op, operands, rewriter,
                     [loc](OpBuilder &builder, ValueRange memRefOperands,
                           ValueRange loopIvs) {
                       typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                       auto loadedLhs = builder.create<AffineLoadOp>(
                               loc, binaryAdaptor.getLhs(), loopIvs);
                       auto loadedRhs = builder.create<AffineLoadOp>(
                               loc, binaryAdaptor.getRhs(), loopIvs);

                       return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                              loadedRhs);
                     });
      return success();
    }
  };

  using AddOpLowering = BinaryOpLowering<flow::AddOp, arith::AddFOp>;
  using SubOpLowering = BinaryOpLowering<flow::SubOp, arith::SubFOp>;
  using MulOpLowering = BinaryOpLowering<flow::MulOp, arith::MulFOp>;
  using DivOpLowering = BinaryOpLowering<flow::DivOp, arith::DivFOp>;

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

  struct SumOpLowering : public ConversionPattern {
    SumOpLowering(MLIRContext *context)
        : ConversionPattern(flow::SumOp::getOperationName(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final {
      llvm::errs() << "sum op lowering\n";
      auto loc = op->getLoc();
      /// %0 = flow.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
      /// %2 = flow.sum %0: tensor<4xf64> to f64

      auto input = operands[0];
      input.dump();
      auto inputType = input.getType().cast<mlir::ShapedType>();// memref<fxf64>
      inputType.dump();
      auto outputType = op->getResult(0).getType();// f64
      outputType.dump();

      auto shape = inputType.getShape();// [4]
      assert(shape.size() == 1 && "expected 1D tensor");
      auto rank = shape.size();


      rewriter.eraseOp(op);
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
  patterns.add<FuncOpLowering, ReturnOpLowering, ConstantOpLowering,
               PrintOpLowering,
               AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
               SumOpLowering>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::flow::createLowerToAffinePass() {
  return std::make_unique<FlowToAffineLowingPass>();
}
