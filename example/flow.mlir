flow.func @main() {
    // %0 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %1 = flow.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %2 = flow.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>

    // %2 = flow.add %1, %0: tensor<2x3xf64>
    // %3 = flow.mul %1, %0: tensor<2x3xf64>
    // %4 = flow.sub %3, %2: tensor<2x3xf64>
    // %5 = flow.div %4, %3: tensor<2x3xf64>
    // flow.print %5: tensor<2x3xf64>

    // %2 = flow.sum %0: tensor<6xf64> to f64
    // %3 = flow.dot %0, %0: tensor<6xf64>, tensor<6xf64> to f64
    // %4 = flow.absf %3: f64
    // %5 = flow.sqrt %4: f64
    // %6 = flow.exp %5: f64
    // %7 = flow.pow %5, %5: f64, f64
    // %8 = flow.log %7: f64
    // %9 = flow.log %8: f64
    // flow.print %9: f64

    // %11 = arith.constant 13.3 : f64
    // %22 = flow.splat %11 : (f64) -> tensor<128xf64>

    // flow.print %2: tensor<3x2xf64>

    // %1 = flow.reshape(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    // %2 = flow.reshape(%1 : tensor<3x2xf64>) to tensor<3x2xf64>
    %2 = flow.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %0 = flow.constant dense<[1.0, 2.0, 3.0, 4.0, 9.0, 32.0]> : tensor<6xf64>
    %1 = flow.get_program_id {axis = 0: i32} : i32
    %3 = flow.sum %0: tensor<6xf64> to f64
    flow.print %3: f64
    flow.print %1: i32
    flow.return
}

