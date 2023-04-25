//
// Created by lay on 2023/4/25.
//

#ifndef FLOW_TRAITS_H
#define FLOW_TRAITS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>

namespace mlir {
  namespace OpTrait {
    namespace impl {
      int constexpr maxTensorNumElements = 1048576;
      LogicalResult verifyTensorSize(Operation *op);
      LogicalResult verifySameOperandsEncoding(Operation *op,
                                               bool allowTensorPointerType = false);
      LogicalResult
      verifySameOperandsAndResultEncoding(Operation *op,
                                          bool allowTensorPointerType = false);

      LogicalResult verifySameLoadStoreOperandsShape(Operation *op);

      LogicalResult verifySameLoadStoreOperandsAndResultShape(Operation *op);

      bool verifyLoadStorePointerAndValueType(Type valueType, Type ptrType);
    }// namespace impl


    template<class ConcreteType>
    class TensorSizeTrait : public TraitBase<ConcreteType, TensorSizeTrait> {
  public:
      static LogicalResult verifyTrait(Operation *op) {
        return impl::verifyTensorSize(op);
      }
    };

    template<typename ConcreteType>
    class SameOperandsAndResultEncoding
        : public TraitBase<ConcreteType, SameOperandsAndResultEncoding> {
  public:
      static LogicalResult verifyTrait(Operation *op) {
        return impl::verifySameOperandsAndResultEncoding(op);
      }
    };

    template<typename ConcreteType>
    class SameOperandsEncoding
        : public TraitBase<ConcreteType, SameOperandsEncoding> {
  public:
      static LogicalResult verifyTrait(Operation *op) {
        return impl::verifySameOperandsEncoding(op);
      }
    };

  }// namespace OpTrait
}// namespace mlir


#endif//FLOW_TRAITS_H
