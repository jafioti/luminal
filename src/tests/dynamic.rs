// Tests with dynamic dimensions

use dfdx::{
    nn::BuildOnDevice,
    prelude::Module as DfdxModule,
    shapes::Rank2,
    tensor::{Cpu, TensorFrom, TensorFromVec},
    tensor_ops::{PermuteTo, ReshapeTo, TryMatMul},
};

use crate::{
    nn::{activation::ReLU, linear::Linear},
    prelude::*,
};

use super::{assert_close, assert_close_data};

#[test]
fn test_movement() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(Const<3>, usize, Const<2>)>("Input");
    let b = a
        .dyn_reshape::<(Const<6>, usize)>(vec![ReshapeDim::Const(6), ReshapeDim::PrevDim(1)])
        .permute::<_, Axes2<1, 0>>();

    a.set_dyn(
        vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
        vec![3, 2, 2],
    );
    b.mark();

    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([
        [[1., 2.], [3., 1.]],
        [[2., 3.], [1., 2.]],
        [[3., 1.], [2., 3.]],
    ]);
    let d_b = d_a.reshape::<Rank2<6, 2>>().permute::<Rank2<2, 6>, _>();

    assert_close_data(
        &b.retrieve().unwrap().real_data(b.view().unwrap()).unwrap(),
        &d_b.as_vec(),
    );
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(usize, Const<3>)>("Input");
    a.set_dyn(vec![1., 2., 3., 1., 2., 3.], vec![2, 3]);
    let b = cx.new_tensor::<R2<3, 3>>("Input");
    b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let c = a.matmul(b);
    c.mark();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b = d_dev.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
    let d_c = d_a.matmul(d_b);

    let r = c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap();
    assert_close_data(&r, &d_c.as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(usize, usize, Const<2>)>("Input");
    a.set_dyn(
        vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
        vec![2, 3, 2],
    );
    let b = cx.new_tensor::<R2<2, 4>>("Input");
    b.set(vec![1., 2., 3., 1., 1., 2., 3., 1.]);
    let c = a.matmul(b);
    c.mark();

    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([
        [[1., 2.], [3., 1.], [2., 3.]],
        [[1., 2.], [3., 1.], [2., 3.]],
    ]);
    let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 3., 1.]]);
    let d_c = d_a.matmul(d_b);

    assert_close_data(
        &c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap(),
        &d_c.as_vec(),
    );
}

#[test]
fn test_feedforward() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let batch = cx.new_tensor::<(usize, Const<3>)>("Input");
    let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
    model
        .0
        .weight
        .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
    let batch_out = model.forward(batch);

    batch.set_dyn(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    batch_out.mark();

    cx.execute();

    let (unoptimized_batch_out, unoptimized_batch_out_view) = (
        batch_out.retrieve().unwrap(),
        batch_out.view().unwrap().clone(),
    );

    cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
    cx.execute();

    assert_close(&unoptimized_batch_out, &batch_out.retrieve().unwrap());

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

    let r = unoptimized_batch_out
        .real_data(&unoptimized_batch_out_view)
        .unwrap();
    assert_close_data(&r, &out.as_vec());
}
