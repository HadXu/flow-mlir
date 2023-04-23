add_mlir_dialect(FlowOps flow)


set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(Combine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)