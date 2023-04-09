module {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @printF64(f64)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(6 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.null : !llvm.ptr<f64>
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %4 = llvm.ptrtoint %3 : !llvm.ptr<f64> to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr<i8>
    %6 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.insertvalue %10, %9[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %0, %11[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %1, %12[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(2 : index) : i64
    %17 = llvm.mlir.constant(3 : index) : i64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(5 : index) : i64
    %20 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %21 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.getelementptr %21[%14] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %20, %22 : !llvm.ptr<f64>
    %23 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %24 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.getelementptr %24[%15] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %23, %25 : !llvm.ptr<f64>
    %26 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %27 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr %27[%16] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %26, %28 : !llvm.ptr<f64>
    %29 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %30 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.getelementptr %30[%17] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %29, %31 : !llvm.ptr<f64>
    %32 = llvm.mlir.constant(9.000000e+00 : f64) : f64
    %33 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.getelementptr %33[%18] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %32, %34 : !llvm.ptr<f64>
    %35 = llvm.mlir.constant(3.200000e+01 : f64) : f64
    %36 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.getelementptr %36[%19] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %35, %37 : !llvm.ptr<f64>
    %38 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.mlir.constant(6 : index) : i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    llvm.br ^bb1(%39, %42 : i64, f64)
  ^bb1(%43: i64, %44: f64):  // 2 preds: ^bb0, ^bb2
    %45 = llvm.icmp "slt" %43, %40 : i64
    llvm.cond_br %45, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %46 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.getelementptr %46[%43] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %48 = llvm.load %47 : !llvm.ptr<f64>
    %49 = llvm.fadd %44, %48  : f64
    %50 = llvm.add %43, %41  : i64
    llvm.br ^bb1(%50, %49 : i64, f64)
  ^bb3:  // pred: ^bb1
    llvm.call @printF64(%44) : (f64) -> ()
    %51 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.bitcast %51 : !llvm.ptr<f64> to !llvm.ptr<i8>
    llvm.call @free(%52) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}