
#include "Dialect/Flow/FlowDialect.h"

using namespace mlir;
using namespace mlir::flow;

#include "Dialect/Flow/FlowOpsDialect.cpp.inc"


void FlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST

#include "Dialect/Flow/FlowOps.cpp.inc"
          >();
}
