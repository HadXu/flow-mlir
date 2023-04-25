#ifndef FLOWGPU_DIALECT
#define FLOWGPU_DIALECT

include "mlir/IR/OpBase.td"

def FlowGPU_Dialect : Dialect {
  let name = "flow_gpu";

  let cppNamespace = "::mlir::flow::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Flow GPU Dialect.
  }];
}

#endif //FLOWGPU_DIALECT