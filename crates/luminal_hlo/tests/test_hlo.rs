use luminal_hlo::import_hlo;

luminal::test_imports!();

#[test]
fn test_import_hlo() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_add.mlir");

    inputs["arg0"].set([[1., 2.], [3., 4.]]);
    inputs["arg1"].set([[1., 2.], [3., 4.]]);

    cx.execute();

    let expected = [2., 4., 6., 8.];
    assert_close(&inputs["0"].data(), &expected);
}
