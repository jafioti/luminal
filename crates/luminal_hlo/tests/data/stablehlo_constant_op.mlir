module {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<1.690000e+02> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
