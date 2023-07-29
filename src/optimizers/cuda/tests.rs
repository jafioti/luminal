use dfdx::prelude::{Module as DfdxModule, *};
use itertools::Itertools;

use super::CudaOptimizer;
use crate::{
    nn::{activation::ReLU, linear::Linear},
    prelude::{Module, *},
    tests::{assert_close, assert_close_data},
};

#[test]
fn test_log2() {
    // We can't use dfdx because it doesn't implement this op
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.log_2();
    b.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    assert_close_data(
        &b.retrieve().unwrap().real_data().unwrap(),
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
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.exp_2();
    b.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    assert_close_data(
        &b.retrieve().unwrap().real_data().unwrap(),
        &vec![1., 2., 3.]
            .into_iter()
            .map(|i: f32| i.exp2())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_log2exp2() {
    // We can't use dfdx because it doesn't implement this op
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.exp_2().log_2();
    b.mark();

    cx.optimize(<(GenericOptimizer, CudaOptimizer)>::default());
    cx.execute();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &[1., 2., 3.]);
}

#[test]
fn test_recip() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.recip();
    b.mark();
    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.recip();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
}

#[test]
fn test_sin() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.sin();
    b.mark();
    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.sin();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
}

#[test]
fn test_sqrt() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = a.sqrt();
    b.mark();
    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_a.sqrt();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
}

#[test]
fn test_add() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a + b;
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a + d_b;

    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
}

#[test]
fn test_sub() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a - b;
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a - d_b;

    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
}

#[test]
fn test_mul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a * b;
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a * d_b;

    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
}

#[test]
fn test_div() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a / b;
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a / d_b;

    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
}

#[test]
fn test_max() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a.max(b);
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a.maximum(d_b);

    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
}

#[test]
fn test_mod() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>();
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>();
    b.set(vec![1., 2., 3.]);
    let c = a % b;
    c.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    // No dfdx equivalent

    assert_close_data(
        &c.retrieve().unwrap().real_data().unwrap(),
        &[1., 2., 3.]
            .into_iter()
            .zip([1., 2., 3.].into_iter())
            .map(|(a, b)| a % b)
            .collect_vec(),
    );
}

// Reduction op tests

#[test]
fn test_sum_reduce() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<2, 2, 3>>();
    a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let b = a.sum_reduce::<_, crate::prelude::Axis<1>>();
    let c = a.sum_reduce::<_, crate::prelude::Axis<0>>();
    let d = a.sum_reduce::<_, crate::prelude::Axis<2>>();
    b.mark();
    c.mark();
    d.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let d_b = d_a.clone().sum::<_, dfdx::shapes::Axis<1>>();
    let d_c = d_a.clone().sum::<_, dfdx::shapes::Axis<0>>();
    let d_d = d_a.sum::<_, dfdx::shapes::Axis<2>>();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    assert_close_data(&d.retrieve().unwrap().real_data().unwrap(), &d_d.as_vec());
}

#[test]
fn test_max_reduce() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<2, 2, 3>>();
    a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let b = a.max_reduce::<_, crate::prelude::Axis<1>>();
    let c = a.max_reduce::<_, crate::prelude::Axis<0>>();
    let d = a.max_reduce::<_, crate::prelude::Axis<2>>();
    b.mark();
    c.mark();
    d.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let d_b = d_a.clone().max::<_, dfdx::shapes::Axis<1>>();
    let d_c = d_a.clone().max::<_, dfdx::shapes::Axis<0>>();
    let d_d = d_a.max::<_, dfdx::shapes::Axis<2>>();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    assert_close_data(&d.retrieve().unwrap().real_data().unwrap(), &d_d.as_vec());
}

#[test]
fn test_mean_reduce() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R2<2, 3>>();
    a.set(vec![1., 2., 3., 1., 2., 3.]);
    let b = a.mean_reduce::<_, crate::prelude::Axis<1>>();
    b.mark();

    cx.optimize(CudaOptimizer::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
    let d_b = d_a.mean::<_, dfdx::shapes::Axis<1>>();

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
}

#[test]
fn test_relu_and_linear() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let batch = cx.new_tensor::<R2<2, 3>>();
    let a = cx.new_tensor::<R1<3>>();

    let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
    model
        .0
        .weight
        .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
    let b = model.forward(a);
    let batch_out = model.forward(batch);

    a.set(vec![1.0, 2.0, 3.0]);
    batch.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    b.mark();
    batch_out.mark();
    cx.execute();

    let unoptimized_b = b.retrieve().unwrap();
    let unoptimized_batch_out = batch_out.retrieve().unwrap();

    cx.optimize(<(CudaOptimizer, GenericOptimizer)>::default());
    cx.execute();

    assert_close(&unoptimized_b, &b.retrieve().unwrap());
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
    let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
    let out = model.forward(a);

    assert_close_data(&unoptimized_b.real_data().unwrap(), &out.as_vec());
}

#[test]
fn test_transformer_encoder_block() {
    let mut cx = Graph::new();
    let model: crate::nn::transformer::encoder::TransformerEncoderBlock<3, 4, 1> =
        InitModule::initialize(&mut cx);
    model
        .attention
        .w_k
        .weight
        .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
    model
        .attention
        .w_q
        .weight
        .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
    model
        .attention
        .w_v
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
    model
        .attention
        .w_o
        .weight
        .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
    model
        .ff
        .0
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
    model
        .ff
        .2
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

    let a = cx.new_tensor::<(usize, crate::shape::Const<3>)>();
    let b = model.forward(a);

    a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], vec![2, 3]);
    b.mark();

    cx.optimize(<(CudaOptimizer, GenericOptimizer)>::default());
    cx.execute();

    let d_dev = Cpu::default();
    let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
        d_dev.build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
    d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
    d_model.self_attn.w_o.weight = d_dev
        .tensor_from_vec(
            vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        )
        .permute();
    d_model.self_attn.w_k.weight = d_dev
        .tensor_from_vec(
            vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        )
        .permute();
    d_model.self_attn.w_q.weight = d_dev
        .tensor_from_vec(
            vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        )
        .permute();
    d_model.self_attn.w_v.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        )
        .permute();
    d_model.ff.0 .0.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
        )
        .permute();
    d_model.ff.0 .0.bias = d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
    d_model.ff.0 .2.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
            (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
        )
        .permute();
    d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
    d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
    d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
    d_model.norm1.epsilon = 1e-5;
    d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
    d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
    d_model.norm2.epsilon = 1e-5;
    let d_a = d_dev.tensor_from_vec(
        vec![-1., 2., 3., 3., 3., -1.],
        (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
    );
    let d_b = d_model.forward(d_a);

    assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
}
