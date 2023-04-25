//
// Created by lay on 2023/4/25.
//

#include "Dialect/Flow/Traits.h"
#include "mlir/IR/TypeUtilities.h"

#include "Dialect/Flow/FlowTypes.h"

using namespace mlir;

static LogicalResult verifySameEncoding(Type typeA, Type typeB,
                                        bool allowTensorPointerType) {

  auto getEncoding = [=](Type type) -> Attribute {
    auto rankedType = type.dyn_cast<RankedTensorType>();
    return rankedType ? rankedType.getEncoding() : Attribute();
  };

  auto encodingA = getEncoding(typeA);
  auto encodingB = getEncoding(typeB);

  if (!encodingA || !encodingB)
    return success();
  return encodingA == encodingB ? success() : failure();
}

LogicalResult OpTrait::impl::verifySameOperandsEncoding(mlir::Operation *op, bool allowTensorPointerType) {

}
