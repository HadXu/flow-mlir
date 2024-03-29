#ifndef FLOW_OPS
#define FLOW_OPS

include "Dialect/Flow/FlowDialect.td"
include "Dialect/Flow/ShapeInferenceInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Flow_Op<string mnemonic, list<Trait> traits = []> : Op<Flow_Dialect, mnemonic, traits>;

def ConstantOp : Flow_Op<"constant", [Pure]> {
    let summary = "constant";
    let description = [{ Constant operation turns a literal into an SSA value. }];

    let arguments = (ins F64ElementsAttr:$value);

    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), [{
              build($_builder, $_state, value.getType(), value);
        }]>,

        OpBuilder<(ins "double":$value)>
      ];

    let hasVerifier = 1;
}

def FuncOp : Flow_Op<"func", [DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{ "flow.func" }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];
    let extraClassDeclaration = [{
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def ReturnOp : Flow_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{ "flow.return"}];

  let arguments = (ins Variadic<F64Tensor>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

def PrintOp : Flow_Op<"print", [MemoryEffects<[MemWrite]>]>  {
    let summary = "print operation";
    let description = [{ "flow.print" }];
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef, F64, I32Tensor, I64Tensor, I32]>:$input);
    let extraClassDeclaration = [{
        Type getPrintType() {
            return getInput().getType();
        }
    }];
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def AddOp : Flow_Op<"add", [Pure]> {
    let summary = "add operation";
    let description = [{ "flow.add" }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def SubOp : Flow_Op<"sub"> {
    let summary = "sub operation";
    let description = [{ "flow.sub" }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def MulOp : Flow_Op<"mul"> {
    let summary = "mul operation";
    let description = [{ "flow.mul" }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def DivOp : Flow_Op<"div"> {
    let summary = "div operation";
    let description = [{ "flow.div" }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
     ];
}


def SumOp : Flow_Op<"sum"> {
    let summary = "sum operation";
    let description = [{ "flow.sum" }];
    let arguments = (ins F_Tensor:$in);
    let results = (outs F_Type:$out);

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$in)>
    ];

    let extraClassDeclaration = [{
        Type getPrintType() {
            return getIn().getType();
        }
    }];

    let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def DotOp : Flow_Op<"dot"> {
    let summary = "dot operation";
    let description = [{ "flow.dot" }];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64:$out);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `to` type($out)";

    let extraClassDeclaration = [{
        Type getOutType() {
            return getOut().getType();
        }
    }];
}

def AbsfOp : Flow_Op<"absf"> {
    let summary = "absf operation";
    let description = [{ "flow.absf" }];
    let arguments = (ins F64:$in);
    let results = (outs F64:$out);

    let assemblyFormat = "$in attr-dict `:` type($in)";
}

def SqrtOp : Flow_Op<"sqrt"> {
    let summary = "sqrt operation";
    let description = [{ "flow.sqrt" }];
    let arguments = (ins F64:$in);
    let results = (outs F64:$out);

    let assemblyFormat = "$in attr-dict `:` type($in)";
}

def ExpOp : Flow_Op<"exp"> {
    let summary = "exp operation";
    let description = [{ "flow.exp" }];
    let arguments = (ins F64:$in);
    let results = (outs F64:$out);

    let assemblyFormat = "$in attr-dict `:` type($in)";
}

def PowOp : Flow_Op<"pow"> {
    let summary = "pow operation";
    let description = [{ "flow.pow" }];
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def LogOp : Flow_Op<"log"> {
    let summary = "log operation";
    let description = [{ "flow.log" }];
    let arguments = (ins F64:$in);
    let results = (outs F64:$out);

    let assemblyFormat = "$in attr-dict `:` type($in)";
}

def Cast : Flow_Op<"cast", [SameOperandsAndResultShape]> {
    let summary = "shape cast operation";
    let description = [{ cast }];
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def TransposeOp : Flow_Op<"transpose", [Pure]> {
    let summary = "transpose operation";
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
      `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;

    let builders = [
      OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;
}

def ReshapeOp : Flow_Op<"reshape", [Pure]> {
    let summary = "reshape operation";
    let description = [{ "flow.reshape" }];

    let arguments = (ins F64Tensor:$input);
    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let results = (outs StaticShapeTensorOf<[F64]>);

    let hasCanonicalizer = 1;
}

//////// triton

def GetProgramIdOp: Flow_Op<"get_program_id", [Pure]> {
    let summary = "get program id";
    let description = [{ "flow.get_program_id" }];

    let arguments = (ins I32Attr:$axis);
    let results = (outs I32:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}


def IntToPtrOp : Flow_Op<"int_to_ptr", [SameOperandsAndResultShape, Pure]> {
    let summary = "Cast int64 to pointer";

    let arguments = (ins F_I64Like:$from);

    let results = (outs F_PtrLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def PtrToIntOp : Flow_Op<"ptr_to_int", [SameOperandsAndResultShape, Pure]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins F_PtrLike:$from);

    let results = (outs F_I64Like:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def SplatOp : Flow_Op<"splat", [Pure,
                                 SameOperandsAndResultElementType,]> {
    let summary = "splat";

    let arguments = (ins F_Type:$src);

    let results = (outs F_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    // let hasFolder = 1;
}

def MakeRangeOp: Flow_Op<"make_range", [Pure]> {
    let summary = "make range";
    let description = [{ Returns an 1D int32 tensor. Values span from $start to $end (exclusive), with step = 1}];

    let arguments = (ins I32Attr:$start, I32Attr:$end);
    let results = (outs F_IntTensor:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def ViewOp : Flow_Op<"view", [NoMemoryEffect,
                               SameOperandsAndResultElementType]> {
    let summary = "view";

    let arguments = (ins F_Tensor:$src);

    let results = (outs F_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    // let hasCanonicalizeMethod = 1;
    // let hasFolder = 1;
}


#endif // FLOW_OPS