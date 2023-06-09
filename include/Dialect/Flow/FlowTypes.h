//
// Created by lay on 2023/4/24.
//

#ifndef FLOW_FLOWTYPES_H
#define FLOW_FLOWTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Flow/FlowOpsTypes.h.inc"

namespace mlir::flow {
  Type getPointeeType(Type type);
  Type getPointerType(Type type);
}// namespace mlir::flow

#endif//FLOW_FLOWTYPES_H
