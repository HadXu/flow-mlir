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

class Flow_Op<string mnemonic, list<Trait> traits = []> : Op<Flow_Dialect, mnemonic, traits>;

def F_Float: AnyTypeOf<[F8E4M3FN, F8E5M2, F16, BF16, F32, F64], "floating-point">;
def F_FloatTensor: TensorOf<[F_Float]>;
def F_FloatLike: AnyTypeOf<[F_Float, F_FloatTensor]>;

def F_Int: AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def F_IntTensor: TensorOf<[F_Int]>;
def F_IntLike: AnyTypeOf<[F_Int, F_IntTensor]>;

def F_I32Like: AnyTypeOf<[I32, I32Tensor]>;
def F_I64Like: AnyTypeOf<[I64, I64Tensor]>;

def F_Type: AnyTypeOf<[F_FloatLike, F_IntLike]>;

#endif // FLOW_DIALECT