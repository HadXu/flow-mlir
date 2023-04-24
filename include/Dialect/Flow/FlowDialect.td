#ifndef FLOW_DIALECT
#define FLOW_DIALECT

include "mlir/IR/OpBase.td"

def Flow_Dialect : Dialect {
    let name = "flow";
    let cppNamespace = "::mlir::flow";
    let useFoldAPI = kEmitFoldAdaptorFolder;
    let dependentDialects = [
        "arith::ArithDialect",
        "math::MathDialect",
        "scf::SCFDialect",
        "cf::ControlFlowDialect"
    ];
}


include "Dialect/Flow/FlowTypes.td"

#endif // FLOW_DIALECT