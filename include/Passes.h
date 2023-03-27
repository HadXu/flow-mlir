//
// Created by lei on 2023/3/27.
//

#ifndef FLOW_PASSES_H
#define FLOW_PASSES_H

#include <memory>

namespace mlir {
  class Pass;

  namespace flow {
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    std::unique_ptr<mlir::Pass> createLowerToAffinePass();
  }// namespace flow
}// namespace mlir

#endif//FLOW_PASSES_H
