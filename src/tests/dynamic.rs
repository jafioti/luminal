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
