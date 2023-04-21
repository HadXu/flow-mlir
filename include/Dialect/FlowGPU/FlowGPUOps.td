#ifndef FLOWGPU_OPS
#define FLOWGPU_OPS

include "mlir/IR/OpBase.td"
include "Dialect/FlowGPU/FlowGPUDialect.td"

class FlowGPUG_Op<string mnemonic, list<Trait> traits = []> : Op<FlowGPU_Dialect, mnemonic, traits>;

#endif //FLOWGPU_OPS