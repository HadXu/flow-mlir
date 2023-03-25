
#include "FlowDialect.h"

using namespace mlir;
using namespace mlir::flow;

#include "FlowOpsDialect.cpp.inc"


void FlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST

#include "FlowOps.cpp.inc"
          >();
}
