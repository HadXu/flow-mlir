# A MLIR Demo Dialect - Flow Dialect

> 关于MLIR 官方的toy例子就非常好，但是非常分散，在实际项目中需要有一个统一的文件组织进行MLIR编译器的开发，本项目基于toy进行修改，统一文件排序规范以及链接库的整理。

## Flow Dialect Ops

1. flow.constant ✅
2. flow.func ✅
3. flow.return ✅
4. flow.transpose
5. flow.reshape
4. flow.add
5. flow.mul
6. flow.sub
7. flow.div
8. flow.print ✅
9. flow.conv
10. flow.dot

## lowing to llvm

主要是flow.print lowing到llvm层面，调用了**printf**方法

## Usage

```bash
cd flow-mlir && mkdir build
cmake ..
make
```










