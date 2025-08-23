use luminal_hlo::import_hlo;

luminal::test_imports!();

#[test]
fn test_stablehlo_binary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_binary_ops.mlir");

    inputs["%arg0"].set([[9., 8.], [7., 6.]]);
    inputs["%arg1"].set([[2., 2.], [3., 4.]]);

    cx.execute();

    let expected = [2., 2., 3., 4., 2., 2., 3., 4.];
    assert_close(&inputs["%8"].data(), &expected);
}

#[test]
fn test_stablehlo_unary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_unary_ops.mlir");

    inputs["%arg0"].set([[49., 64.], [25., 36.]]);

    cx.execute();

    let expected = [7., 8., 5., 6.];
    assert_close(&inputs["%4"].data(), &expected);
}

#[test]
fn test_stablehlo_constant_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_constant_op.mlir");

    inputs["%arg0"].set([1., 1., 1., 1.]);

    cx.execute();

    let expected = [169., 169., 169., 169.];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_broadcast_in_dim_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_broadcast_in_dim_op.mlir");

    inputs["%arg0"].set([1., 1., 1., 1.]);
    inputs["%arg1"].set([1., 1., 1., 1.]);

    cx.execute();

    let expected = [142., 142., 142., 142., 142., 142., 142., 142.];
    assert_close(&inputs["%2"].data(), &expected);
}

// #[test]
// fn test_parse_tensor_shape() {
//     // Test regular tensor shapes
//     assert_eq!(parse_tensor_shape("tensor<1x16x1x1xf32>"), vec![1, 16, 1, 1]);
//     assert_eq!(parse_tensor_shape("tensor<96x3x7x7xf32>"), vec![96, 3, 7, 7]);
//     assert_eq!(parse_tensor_shape("tensor<1x1000xf32>"), vec![1, 1000]);
        
//     // Test scalar tensor
//     assert_eq!(parse_tensor_shape("tensor<f32>"), vec![1]);
        
//     // Test single dimension
//     assert_eq!(parse_tensor_shape("tensor<16xf32>"), vec![16]);
// }

// #[test]
// fn test_parse_output_shape_from_op() {
//     // Test the example from the user
//     let op_line = "%8 = stablehlo.reshape %arg3 : (tensor<16xf32>) -> tensor<1x16x1x1xf32>";
//     assert_eq!(parse_output_shape_from_op(op_line), vec![1, 16, 1, 1]);
        
//     // Test other examples
//     let op_line2 = "%0 = stablehlo.convolution(%arg52, %arg0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x224x224xf32>, tensor<96x3x7x7xf32>) -> tensor<1x96x109x109xf32>";
//     assert_eq!(parse_output_shape_from_op(op_line2), vec![1, 96, 109, 109]);
        
//     // Test scalar output
//     let op_line3 = "%cst = stablehlo.constant dense<1.690000e+02> : tensor<f32>";
//     assert_eq!(parse_output_shape_from_op(op_line3), vec![1]);
// }
