#include "FlowDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::flow;

namespace {
  static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
      return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
      if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                 result.operands))
        return mlir::failure();
      result.addTypes(funcType.getResults());
      return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
      return mlir::failure();
    result.addTypes(type);
    return mlir::success();
  }

  static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    Type resultType = *op->result_type_begin();

    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; })) {
      printer << resultType;
      return;
    }

    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
  }
}// namespace


void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

mlir::LogicalResult ConstantOp::verify() {
  auto resultType = getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  auto attrType = getValue().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                     "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  auto buildFuncType =
          [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
             llvm::ArrayRef<mlir::Type> results,
             mlir::function_interface_impl::VariadicFlag,
             std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
          parser, result, /*allowVariadic=*/false,
          getFunctionTypeAttrName(result.name), buildFuncType,
          getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
          p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
          getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::Region *FuncOp::getCallableRegion() { return &getBody(); }

llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

mlir::LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

void AddOp::print(mlir::OpAsmPrinter &p) {
  printBinaryOp(p, *this);
}

mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void SubOp::print(mlir::OpAsmPrinter &p) {
  printBinaryOp(p, *this);
}

mlir::ParseResult SubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

#define GET_OP_CLASSES
#include "FlowOps.cpp.inc"
