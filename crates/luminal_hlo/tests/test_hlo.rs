use luminal_hlo::import_hlo;

luminal::test_imports!();

#[test]
fn test_stablehlo_binary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_binary_ops.mlir");

    inputs["arg0"].set([[9., 8.], [7., 6.]]);
    inputs["arg1"].set([[2., 2.], [3., 4.]]);

    cx.execute();

    let expected = [1., 0., 1., 2.];
    assert_close(&inputs["4"].data(), &expected);
}
