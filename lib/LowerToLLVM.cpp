//
// Created by lei on 2023/3/27.
//


#include "FlowDialect.h"
#include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;


namespace {
  class PrintOpLowering : public ConversionPattern {
public:
    explicit PrintOpLowering(MLIRContext *context)
        : ConversionPattern(flow::PrintOp::getOperationName(), 1, context) {}
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override {
      /// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L1442
      /// https://github.com/llvm/llvm-project/blob/release/16.x/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td#L2437
      auto parent = op->getParentOfType<ModuleOp>();
      auto type = op->getOperandTypes().front();

      if (type.isF64()) {
        Operation *printer = LLVM::lookupOrCreatePrintF64Fn(parent);
        rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange(),
                                      SymbolRefAttr::get(printer),
                                      ValueRange({operands, }));
        rewriter.eraseOp(op);
        return success();
      }

      auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
      auto memRefShape = memRefType.getShape();
      auto loc = op->getLoc();

      auto parentModule = op->getParentOfType<ModuleOp>();
      auto printfRef = getOrInsertPrintf(rewriter, parentModule);


      Value formatSpecifierCst = getOrCreateGlobalString(
              loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
      Value newLineCst = getOrCreateGlobalString(
              loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

      SmallVector<Value, 4> loopIvs;
      for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
        auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto upperBound =
                rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto loop =
                rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        for (Operation &nested: *loop.getBody())
          rewriter.eraseOp(&nested);
        loopIvs.push_back(loop.getInductionVar());

        // Terminate the loop body.
        rewriter.setInsertionPointToEnd(loop.getBody());

        // Insert a newline after each of the inner dimensions of the shape.
        if (i != e - 1)
          rewriter.create<func::CallOp>(loc, printfRef,
                                        rewriter.getIntegerType(32), newLineCst);
        rewriter.create<scf::YieldOp>(loc);
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      // Generate a call to printf for the current element of the loop.
      auto printOp = cast<flow::PrintOp>(op);
      auto elementLoad =
              rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
      rewriter.create<func::CallOp>(
              loc, printfRef, rewriter.getIntegerType(32),
              ArrayRef<Value>({formatSpecifierCst, elementLoad}));

      // Notify the rewriter that this operation has been removed.
      rewriter.eraseOp(op);
      return success();
    }

private:
    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                               ModuleOp module) {
      auto *context = module.getContext();
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
        return SymbolRefAttr::get(context, "printf");
      }

      auto llvmI32Ty = IntegerType::get(context, 32);
      auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

      // Insert the printf function into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
      return SymbolRefAttr::get(context, "printf");
    }

    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                         StringRef name, StringRef value,
                                         ModuleOp module) {
      // Create the global at the entry of the module.
      LLVM::GlobalOp global;
      if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
        OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
        global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                LLVM::Linkage::Internal, name,
                                                builder.getStringAttr(value),
                                                /*alignment=*/0);
      }

      // Get the pointer to the first character in the global string.
      Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
      Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                    builder.getIndexAttr(0));
      return builder.create<LLVM::GEPOp>(
              loc,
              LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
              globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
  };
}// namespace

namespace {
  struct FlowToLLVMLowingPass : public PassWrapper<FlowToLLVMLowingPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlowToLLVMLowingPass)
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    void runOnOperation() final;
  };
}// namespace

void FlowToLLVMLowingPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  patterns.add<PrintOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::flow::createLowerToLLVMPass() {
  return std::make_unique<FlowToLLVMLowingPass>();
}