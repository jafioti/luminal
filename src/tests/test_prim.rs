use crate::{prelude::*, tests::assert_close};
use dfdx::prelude::*;
use itertools::Itertools;

// Movement op tests

#[test]
fn test_reshape() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3)).set([[1., 2., 3.], [1., 2., 3.]]);
    let b = a.reshape(6).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b: dfdx::tensor::Tensor<Rank1<6>, f32, Cpu> = d_a.reshape();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_permute() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3)).set([[1., 2., 3.], [1., 2., 3.]]);
    let b = a.permute((1, 0)).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.permute();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_expand() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.expand(1, 2).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b: dfdx::tensor::Tensor<Rank2<3, 2>, f32, Cpu> = d_a.broadcast();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_slice() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3)).set([[1., 2., 3.], [1., 2., 3.]]);
    let b = a.slice((1.., ..)).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b = d_a.slice((1.., ..));

    assert_close(&b.data(), &d_b.as_vec());
}

// Unary op tests

#[test]
fn test_log2() {
    // We can't use dfdx because it doesn't implement this op
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.log2().retrieve();
    cx.execute();

    assert_close(
        &b.data(),
        &vec![1., 2., 3.]
            .into_iter()
            .map(|i: f32| i.log2())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_exp2() {
    // We can't use dfdx because it doesn't implement this op
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.exp2().retrieve();
    cx.execute();

    assert_close(
        &b.data(),
        &vec![1., 2., 3.]
            .into_iter()
            .map(|i: f32| i.exp2())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_reciprocal() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.reciprocal().retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.recip();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_sin() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.sin().retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.sin();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_sqrt() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = a.sqrt().retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.sqrt();

    assert_close(&b.data(), &d_b.as_vec());
}

// Binary op tests

#[test]
fn test_add() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([1., 2., 3.]);
    let c = (a + b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a + d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_sub() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([1., 2., 3.]);
    let c = (a - b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a - d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_mul() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([1., 2., 3.]);
    let c = (a * b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a * d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_permute_mul() {
    let mut cx = Graph::new();
    let a = cx.tensor((3, 2)).set([[1., 2.], [3., 2.], [3., 1.]]);
    let b = cx.tensor((3, 2)).set([[1., 2.], [3., -1.], [3., 0.]]);
    let c = a.expand(2, 3) * b.expand(2, 3);
    c.retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2.], [3., 2.], [3., 1.]]);
    let d_b = d_dev.tensor([[1., 2.], [3., -1.], [3., 0.]]);
    let d_c = d_a.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>()
        * d_b.broadcast::<Rank3<3, 2, 3>, dfdx::prelude::Axis<2>>();

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_div() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([1., 2., 3.]);
    let c = (a / b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a / d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_max() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 0., 3.]);
    let b = cx.tensor(3).set([1., 2., -2.]);
    let c = a.maximum(b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 0., 3.]);
    let d_b = d_dev.tensor([1., 2., -2.]);
    let d_c = d_a.maximum(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_mod() {
    let mut cx = Graph::new();
    let a = cx.tensor(3).set([1., 2., 3.]);
    let b = cx.tensor(3).set([1., 2., 3.]);
    let c = (a % b).retrieve();
    cx.execute();

    // No dfdx equivalent

    assert_close(
        &c.data(),
        &[1., 2., 3.]
            .into_iter()
            .zip([1., 2., 3.])
            .map(|(a, b)| a % b)
            .collect_vec(),
    );
}

// Reduction op tests

#[test]
fn test_sum() {
    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let b = a.sum(1).retrieve();
    let c = a.sum(0).retrieve();
    let d = a.sum(2).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let d_b = d_a.clone().sum::<_, dfdx::shapes::Axis<1>>();
    let d_c = d_a.clone().sum::<_, dfdx::shapes::Axis<0>>();
    let d_d = d_a.sum::<_, dfdx::shapes::Axis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_sum2() {
    let mut cx = Graph::new();
    let a = cx.tensor((1, 2, 2, 3)).set([[
        [[34.4, -96.0, 144.0], [43.0, 560.0, 180.0]],
        [[39.6, -120.0, 180.0], [49.5, 700.0, 225.0]],
    ]]);
    let b = a.sum(3).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[
        [[34.4, -96.0, 144.0], [43.0, 560.0, 180.0]],
        [[39.6, -120.0, 180.0], [49.5, 700.0, 225.0]],
    ]]);
    let d_b = d_a.sum::<_, dfdx::shapes::Axis<3>>();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_max_reduce() {
    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let b = a.max(1).retrieve();
    let c = a.max(0).retrieve();
    let d = a.max(2).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let d_b = d_a.clone().max::<_, dfdx::shapes::Axis<1>>();
    let d_c = d_a.clone().max::<_, dfdx::shapes::Axis<0>>();
    let d_d = d_a.max::<_, dfdx::shapes::Axis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_dot() {
    let mut cx = Graph::new();
    let a = cx.tensor(5).set([34.4, -96.0, 144.0, 43.0, 560.0]);
    let b = cx.tensor(5).set([43.0, 560.0, 180.0, 700.0, 225.0]);
    let c = a.dot(b).retrieve();
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([34.4, -96.0, 144.0, 43.0, 560.0]);
    let d_b = d_dev.tensor([43.0, 560.0, 180.0, 700.0, 225.0]);
    let d_c = (d_a * d_b).sum();
    assert_close(&c.data(), &d_c.as_vec());
}
