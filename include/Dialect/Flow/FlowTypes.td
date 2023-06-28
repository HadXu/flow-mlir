#ifndef FLOW_TYPES
#define FLOW_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/Flow/FlowDialect.td"

class FlowTypeDef<string name, string _mnemonic> : TypeDef<Flow_Dialect, name> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

def My_IntegerType : FlowTypeDef<"Integer", "int"> {
    let summary = "Integer type with arbitrary precision up to a fixed limit";
      let description = [{
        Integer types have a designated bit width.
      }];

    /// Here we defined a single parameter for the type, which is the bitwidth.
    let parameters = (ins "unsigned":$width);

    let assemblyFormat = "`<` $width `>`";
    let genVerifyDecl = 1;
}

def F_Float: AnyTypeOf<[F8E4M3FN, F8E5M2, F16, BF16, F32, F64], "floating-point">;
def F_FloatTensor: TensorOf<[F_Float]>;
def F_FloatLike: AnyTypeOf<[F_Float, F_FloatTensor]>;

def F_Int: AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def F_IntTensor: TensorOf<[F_Int]>;
def F_IntLike: AnyTypeOf<[F_Int, F_IntTensor]>;

def F_I32Like: AnyTypeOf<[I32, I32Tensor]>;
def F_I64Like: AnyTypeOf<[I64, I64Tensor]>;

class F_PtrOf<list<Type> pointeeTypes> :
    DialectType<Flow_Dialect,
                And<[CPred<"$_self.isa<::mlir::flow::PointerType>()">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }($_self.cast<::mlir::flow::PointerType>().getPointeeType())">]>,
                "ptr", "::mlir::flow::PointerType">;


def F_PtrType : FlowTypeDef<"Pointer", "ptr"> {
    let summary = "Pointer type (`::mlir::flow::PointerType`) in Triton IR type system";

    let description = [{
        Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

def F_Ptr : F_PtrOf<[AnyType]>;

// Tensor of Pointer Type
def F_PtrTensor : TensorOf<[F_Ptr]>;

// Tensor of Pointer Type or Pointer type: `tensor<ptr<>>` or `ptr<>`
def F_PtrLike : AnyTypeOf<[F_Ptr, F_PtrTensor]>;

// Tensor Type
def F_FpIntTensor : AnyTypeOf<[F_FloatTensor, F_IntTensor]>;
def F_Tensor : AnyTypeOf<[F_FpIntTensor, F_PtrTensor]>;

// Pointer Type to Tensor Type: `ptr<tensor<>>`
def F_TensorPtr : F_PtrOf<[F_Tensor]>;

// Any Type in Triton IR
def F_Type : AnyTypeOf<[F_FloatLike, F_IntLike, F_PtrLike, F_TensorPtr]>;

#endif //FLOW_TYPES