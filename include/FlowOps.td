#ifndef FLOW_OPS
#define FLOW_OPS

include "FlowDialect.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def PrintOp : Flow_Op<"print"> {
    let summary = "print operation";
    let description = [{ "flow.print" }];
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef, F64]>:$input);
    let extraClassDeclaration = [{
        Type getPrintType() {
            return getInput().getType();
        }
    }];
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def AddOp : Flow_Op<"add"> {
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
    let arguments = (ins F64Tensor:$in);
    let results = (outs F64:$out);

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



#endif // FLOW_OPS