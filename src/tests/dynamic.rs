// Tests with dynamic dimensions

use dfdx::{
    nn::BuildOnDevice,
    prelude::Module as DfdxModule,
    shapes::Rank2,
    tensor::{Cpu, TensorFrom, TensorFromVec},
    tensor_ops::{PermuteTo, ReshapeTo, TryMatMul},
};

use crate::prelude::*;

use super::assert_close;

#[test]
fn test_movement() {
    let mut cx = Graph::new();
    let a = cx.tensor::<(Const<3>, Dyn<'a'>, Const<2>)>().set_dyn(
        vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
        &[3, 2, 2],
    );
    let b = a
        .reshape::<(Const<6>, Dyn<'a'>)>()
        .permute::<_, Axes2<1, 0>>()
        .keep()
        .retrieve();

    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([
        [[1., 2.], [3., 1.]],
        [[2., 3.], [1., 2.]],
        [[3., 1.], [2., 3.]],
    ]);
    let d_b = d_a.reshape::<Rank2<6, 2>>().permute::<Rank2<2, 6>, _>();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a = cx
        .tensor::<(Dyn<'a'>, Const<3>)>()
        .set_dyn(vec![1., 2., 3., 1., 2., 3.], &[2, 3]);
    let b = cx
        .tensor::<R2<3, 3>>()
        .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let c = a.matmul(b).retrieve();

    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b = d_dev.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
    let d_c = d_a.matmul(d_b);

    let r = c.data();
    assert_close(&r, &d_c.as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let a = cx.tensor::<(Dyn<'a'>, Dyn<'b'>, Const<2>)>().set_dyn(
        vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
        &[2, 3, 2],
    );
    let b = cx
        .tensor::<R2<2, 4>>()
        .set(vec![1., 2., 3., 1., 1., 2., 3., 1.]);
    let c = a.matmul(b).retrieve();

    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([
        [[1., 2.], [3., 1.], [2., 3.]],
        [[1., 2.], [3., 1.], [2., 3.]],
    ]);
    let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 3., 1.]]);
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_feedforward() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let batch = cx
        .tensor::<(Dyn<'a'>, Const<3>)>()
        .set_dyn(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
    model
        .0
        .weight
        .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
    let mut batch_out = model.forward(batch).retrieve();

    cx.execute();

    let unoptimized_batch_out = batch_out.data();
    batch_out.drop();

    cx.compile(<CPUCompiler>::default(), &mut batch_out);
    cx.execute();
    assert_close(&unoptimized_batch_out, &batch_out.data());

    // Test against dfdx
    let dev = Cpu::default();
    let mut model = <(
        dfdx::nn::modules::builders::UnbiasedLinear<3, 4>,
        dfdx::nn::modules::builders::ReLU,
        dfdx::nn::modules::builders::UnbiasedLinear<4, 2>,
    )>::build_on_device(&dev);
    // Set weights
    model.0.weight = dev
        .tensor_from_vec(
            vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
        )
        .permute();
    model.2.weight = dev
        .tensor_from_vec(
            vec![1., 2., 3., 1., 2., 3., 1., 2.],
            (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<2>),
        )
        .permute();
    let a = dev.tensor_from_vec(
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
    );
    let out = model.forward(a);

    assert_close(&unoptimized_batch_out, &out.as_vec());
}
