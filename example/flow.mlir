flow.func @main() {
    // %0 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %1 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %2 = flow.add %1, %0: tensor<2x3xf64>
    // %3 = flow.mul %1, %0: tensor<2x3xf64>
    // %4 = flow.sub %3, %2: tensor<2x3xf64>
    // %5 = flow.div %4, %3: tensor<2x3xf64>
    // flow.print %5: tensor<2x3xf64>
    %0 = flow.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
    %2 = flow.sum %0: f64
    flow.return
}