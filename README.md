# A MLIR Demo Dialect - Flow Dialect (on LLVM(release/16.x))

> 关于MLIR 官方的toy例子就非常好，但是非常分散，在实际项目中需要有一个统一的文件组织进行MLIR编译器的开发，本项目基于toy进行修改，统一文件排序规范以及链接库的整理。

## Flow Dialect Ops

1. flow.constant ✅
2. flow.func ✅
3. flow.return ✅
4. flow.transpose
5. flow.reshape
4. flow.add ✅
5. flow.mul ✅
6. flow.sub ✅
7. flow.div ✅
8. flow.print ✅ (支持各种print)
9. flow.conv
10. flow.dot ✅
11. flow.sum ✅ (1D)
12. support math Dialect ✅

## 1. flow dialect
```c++
flow.func @main() {
    %0 = flow.constant dense<[1.0, 2.0, 3.0, 4.0, 9.0, 32.0]> : tensor<6xf64>
    %2 = flow.sum %0: tensor<6xf64> to f64
    %3 = flow.dot %0, %0: tensor<6xf64>, tensor<6xf64> to f64
    %4 = math.atan %3: f64
    flow.print %4: f64
    flow.return
}
```


## 2. lowing to affine

## 3. lowing to llvm

## 4. run with jit 
use 
```c++
mlir::ExecutionEngineOptions engineOptions;
engineOptions.transformer = optPipeline;
engineOptions.sharedLibPaths = {"", ""};
```

## lowing to llvm
将custom dialect lowering to vector dialect. (need to read vector lowering source file).

## Usage

```bash
cd flow-mlir && mkdir build
cmake ..
make
```










