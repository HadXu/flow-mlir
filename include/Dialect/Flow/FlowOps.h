#ifndef FLOWOPS_H
#define FLOWOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Flow/ShapeInferenceInterface.h"


#define GET_OP_CLASSES
#include "Dialect/Flow/FlowOps.h.inc"

#endif//FLOWOPS_H
