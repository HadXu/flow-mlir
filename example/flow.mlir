#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>

flow.func @main() {
    // %0 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %1 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %2 = flow.add %1, %0: tensor<2x3xf64>
    // %3 = flow.mul %1, %0: tensor<2x3xf64>
    // %4 = flow.sub %3, %2: tensor<2x3xf64>
    // %5 = flow.div %4, %3: tensor<2x3xf64>
    // flow.print %5: tensor<2x3xf64>
    %0 = flow.constant dense<[1.0, 2.0, 3.0, 4.0, 9.0, 32.0]> : tensor<6xf64>
    %2 = flow.sum %0: tensor<6xf64> to f64
    %3 = flow.dot %0, %0: tensor<6xf64>, tensor<6xf64> to f64
    %4 = flow.absf %3: f64
    %5 = flow.sqrt %4: f64
    flow.print %5: f64

    flow.return
}
