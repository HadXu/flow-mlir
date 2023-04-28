//
// Created by lay on 2023/4/28.
//

#ifndef FLOW_SHAPEINFERENCEINTERFACE_H
#define FLOW_SHAPEINFERENCEINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::flow {
#include "Dialect/Flow/ShapeInferenceOpInterfaces.h.inc"
}// namespace mlir::flow

#endif//FLOW_SHAPEINFERENCEINTERFACE_H
