#ifndef Flow_COMBINE
#define Flow_COMBINE

include "mlir/IR/PatternBase.td"
include "Dialect/Flow/FlowOps.td"

//class Pattern<
//    dag sourcePattern, list<dag> resultPatterns,
//    list<dag> additionalConstraints = [],
//    dag benefitsAdded = (addBenefit 0)>;


// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

// Reshape(Constant(x)) = x'
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // Flow_COMBINE