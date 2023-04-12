#include "FlowDialect.h"
#include "Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <memory>


namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
  enum Action {
    None,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT,
  };
}

static cl::opt<enum Action> emitAction(
        "emit", cl::desc("Select the kind of output"),
        cl::values(clEnumValN(DumpMLIR, "mlir", "out mlir")),
        cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "out mlir-affine")),
        cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "out mlir-llvm")),
        cl::values(clEnumValN(DumpLLVMIR, "llvm", "out llvm")),
        cl::values(clEnumValN(RunJIT, "jit", "run jit")));


int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFile(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return -1;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(&context);
  applyPassManagerCLOptions(pm);

  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (isLoweringToAffine) {
    pm.addPass(mlir::flow::createLowerToAffinePass());
    mlir::OpPassManager &optPM = pm.nest<mlir::flow::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToLLVM) {
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());

    pm.addPass(mlir::flow::createLowerToLLVMPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  engineOptions.sharedLibPaths = {"/home/lay/llvm/build/lib/libmlir_runner_utils.so", "/home/lay/llvm/build/lib/libmlir_c_runner_utils.so"};
  //  engineOptions.sharedLibPaths = {"/Users/lei/soft/llvm-project/build/lib/libmlir_runner_utils.dylib", "/Users/lei/soft/llvm-project/build/lib/libmlir_c_runner_utils.dylib"};

  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  llvm::errs() << *llvmModule << "\n";


  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "flow compiler\n");

  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::flow::FlowDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;

  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  if (emitAction == Action::DumpLLVMIR) {
    return dumpLLVMIR(*module);
  }

  if (emitAction == Action::RunJIT) {
    return runJit(*module);
  }
  llvm::errs() << "No action specified\n";

  return 0;
}