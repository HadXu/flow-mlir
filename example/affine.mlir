module {
  func.func @main() {
    %alloc = memref.alloc() : memref<4xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc[%c0] : memref<4xf64>
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.store %cst_0, %alloc[%c1] : memref<4xf64>
    %cst_1 = arith.constant 3.000000e+00 : f64
    affine.store %cst_1, %alloc[%c2] : memref<4xf64>
    %cst_2 = arith.constant 4.000000e+00 : f64
    affine.store %cst_2, %alloc[%c3] : memref<4xf64>
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c0_4 = arith.constant 0 : index
    %0 = memref.load %alloc[%c0_4] : memref<4xf64>
    %1 = arith.addf %cst_3, %0 : f64
    %c1_5 = arith.constant 1 : index
    %2 = memref.load %alloc[%c1_5] : memref<4xf64>
    %3 = arith.addf %1, %2 : f64
    %c2_6 = arith.constant 2 : index
    %4 = memref.load %alloc[%c2_6] : memref<4xf64>
    %5 = arith.addf %3, %4 : f64
    %c3_7 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_7] : memref<4xf64>
    %7 = arith.addf %5, %6 : f64
    vector.print %7 : f64
    memref.dealloc %alloc : memref<4xf64>
    return
  }
}