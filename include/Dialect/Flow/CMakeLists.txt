add_mlir_dialect(FlowOps flow)


set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(Combine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)