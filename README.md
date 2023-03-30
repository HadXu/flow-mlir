# A MLIR Demo Dialect - Flow Dialect

> 关于MLIR 官方的toy例子就非常好，但是非常分散，在实际项目中需要有一个统一的文件组织进行MLIR编译器的开发，本项目基于toy进行修改，统一文件排序规范以及链接库的整理。

## Flow Dialect Ops

1. flow.constant ✅
2. flow.func ✅
3. flow.return ✅
4. flow.transpose
5. flow.reshape
4. flow.add ✅
5. flow.mul
6. flow.sub ✅
7. flow.div
8. flow.print ✅
9. flow.conv
10. flow.dot

## 1. flow dialect
```c++
flow.func @main() {
    %0 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = flow.add %1, %0: tensor<2x3xf64>
    %3 = flow.sub %1, %0: tensor<2x3xf64>
    flow.print %3: tensor<2x3xf64>
    flow.return
}
```


## 2. lowing to affine
```c++
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<2x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc[%c0, %c0] : memref<2x3xf64>
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.store %cst_0, %alloc[%c0, %c1] : memref<2x3xf64>
    %cst_1 = arith.constant 3.000000e+00 : f64
    affine.store %cst_1, %alloc[%c0, %c2] : memref<2x3xf64>
    %cst_2 = arith.constant 4.000000e+00 : f64
    affine.store %cst_2, %alloc[%c1, %c0] : memref<2x3xf64>
    %cst_3 = arith.constant 5.000000e+00 : f64
    affine.store %cst_3, %alloc[%c1, %c1] : memref<2x3xf64>
    %cst_4 = arith.constant 6.000000e+00 : f64
    affine.store %cst_4, %alloc[%c1, %c2] : memref<2x3xf64>
    flow.print %alloc : memref<2x3xf64>
    memref.dealloc %alloc : memref<2x3xf64>
    return
  }
}
```

## 3. lowing to llvm
```c++
module {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(6 : index) : i64
    %4 = llvm.mlir.null : !llvm.ptr<f64>
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %6 = llvm.ptrtoint %5 : !llvm.ptr<f64> to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr<i8>
    %8 = llvm.bitcast %7 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %0, %13[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %1, %14[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %1, %15[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %2, %16[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(2 : index) : i64
    %21 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %22 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.mlir.constant(3 : index) : i64
    %24 = llvm.mul %18, %23  : i64
    %25 = llvm.add %24, %18  : i64
    %26 = llvm.getelementptr %22[%25] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %21, %26 : !llvm.ptr<f64>
    %27 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %28 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.mlir.constant(3 : index) : i64
    %30 = llvm.mul %18, %29  : i64
    %31 = llvm.add %30, %19  : i64
    %32 = llvm.getelementptr %28[%31] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %27, %32 : !llvm.ptr<f64>
    %33 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %34 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.mlir.constant(3 : index) : i64
    %36 = llvm.mul %18, %35  : i64
    %37 = llvm.add %36, %20  : i64
    %38 = llvm.getelementptr %34[%37] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %33, %38 : !llvm.ptr<f64>
    %39 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %40 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.mlir.constant(3 : index) : i64
    %42 = llvm.mul %19, %41  : i64
    %43 = llvm.add %42, %18  : i64
    %44 = llvm.getelementptr %40[%43] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %39, %44 : !llvm.ptr<f64>
    %45 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %46 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.mlir.constant(3 : index) : i64
    %48 = llvm.mul %19, %47  : i64
    %49 = llvm.add %48, %19  : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %45, %50 : !llvm.ptr<f64>
    %51 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %52 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.mlir.constant(3 : index) : i64
    %54 = llvm.mul %19, %53  : i64
    %55 = llvm.add %54, %20  : i64
    %56 = llvm.getelementptr %52[%55] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %51, %56 : !llvm.ptr<f64>
    %57 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.getelementptr %57[%58, %58] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %60 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
    %61 = llvm.mlir.constant(0 : index) : i64
    %62 = llvm.getelementptr %60[%61, %61] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %63 = llvm.mlir.constant(0 : index) : i64
    %64 = llvm.mlir.constant(2 : index) : i64
    %65 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%63 : i64)
  ^bb1(%66: i64):  // 2 preds: ^bb0, ^bb5
    %67 = llvm.icmp "slt" %66, %64 : i64
    llvm.cond_br %67, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.mlir.constant(3 : index) : i64
    %70 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%68 : i64)
  ^bb3(%71: i64):  // 2 preds: ^bb2, ^bb4
    %72 = llvm.icmp "slt" %71, %69 : i64
    llvm.cond_br %72, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %73 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %74 = llvm.mlir.constant(3 : index) : i64
    %75 = llvm.mul %66, %74  : i64
    %76 = llvm.add %75, %71  : i64
    %77 = llvm.getelementptr %73[%76] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %78 = llvm.load %77 : !llvm.ptr<f64>
    %79 = llvm.call @printf(%59, %78) : (!llvm.ptr<i8>, f64) -> i32
    %80 = llvm.add %71, %70  : i64
    llvm.br ^bb3(%80 : i64)
  ^bb5:  // pred: ^bb3
    %81 = llvm.call @printf(%62) : (!llvm.ptr<i8>) -> i32
    %82 = llvm.add %66, %65  : i64
    llvm.br ^bb1(%82 : i64)
  ^bb6:  // pred: ^bb1
    %83 = llvm.extractvalue %17[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = llvm.bitcast %83 : !llvm.ptr<f64> to !llvm.ptr<i8>
    llvm.call @free(%84) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}
```

## 4. run with jit
```c++
1.000000 2.000000 3.000000
4.000000 5.000000 6.000000
```

## lowing to llvm
主要是flow.print lowing到llvm层面，调用了内置**printf**方法

## Usage

```bash
cd flow-mlir && mkdir build
cmake ..
make
```










