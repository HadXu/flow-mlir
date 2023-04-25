#include "Dialect/Flow/FlowTypes.h"
#include "Dialect/Flow/FlowDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"


using namespace mlir;
using namespace mlir::flow;

#define GET_TYPEDEF_CLASSES
#include "Dialect/Flow/FlowOpsTypes.cpp.inc"

void FlowDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Flow/FlowOpsTypes.cpp.inc"
          >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return {};

  if (parser.parseGreater())
    return {};

  // TODO: also print address space?
  return PointerType::get(pointeeType, 1);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ">";
}


namespace mlir::flow {
  Type getPointeeType(Type type) {
    if (auto tensorTy = type.dyn_cast<RankedTensorType>()) {
      // Tensor of pointers
      auto shape = tensorTy.getShape();
      auto ptrType = tensorTy.getElementType().dyn_cast<PointerType>();
      Type pointeeType = ptrType.getPointeeType();
      return RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
    } else if (auto ptrType = type.dyn_cast<PointerType>()) {
      // scalar pointer
      Type pointeeType = ptrType.getPointeeType();
      return pointeeType;
    }
    return type;
  }

  Type getPointerType(Type type) { return PointerType::get(type, 1); }

}// namespace mlir::flow
