use dfdx::prelude::{Module as DfdxModule, *};
use half::f16;
use itertools::Itertools;
use rand::Rng;

use super::MetalFp32Compiler;
use crate::{
    nn::{activation::ReLU, linear::Linear},
    prelude::{Module, *},
};

crate::test_imports!();

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.new_tensor::<R2<3, 4>>("Input");
    a.set(data.clone());
    let b = a.permute::<R2<4, 3>, _>().reshape::<R2<12, 1>>();
    b.mark();
    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<3>, DConst::<4>));
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_log2() {
    let mut cx = Graph::new();
    let data = random_vec(3);
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(data.clone());
    let b = a.log_2();
    b.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    assert_close(
        &b.data(),
        &data.into_iter().map(|i: f32| i.log2()).collect::<Vec<_>>(),
    );
}

#[test]
fn test_exp2() {
    let mut cx = Graph::new();
    let data = random_vec(3);
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(data.clone());
    let b = a.exp_2();
    b.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    assert_close(
        &b.data(),
        &data.into_iter().map(|i: f32| i.exp2()).collect::<Vec<_>>(),
    );
}

#[test]
fn test_recip() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 4096.]);
    let b = a.recip();
    b.mark();
    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 4096.]).to_dtype::<f16>();
    let d_b = d_a.recip();

    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
}

#[test]
fn test_sin() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = a.sin();
    b.mark();
    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_a.sin();

    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
}

#[test]
fn test_sqrt() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = a.sqrt();
    b.mark();
    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_a.sqrt();

    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
}

#[test]
fn test_add() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a + b;
    c.mark();

    cx.compile(MetalFp32Compiler::default());
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
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a - b;
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a - d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_square() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(Dyn<'b'>, Dyn<'s'>, crate::prelude::Const<4096>)>("Input");
    let mut rng = rand::thread_rng();
    let data = (0..40960)
        .map(|_| rng.gen_range(-0.01..0.01))
        .collect::<Vec<f32>>();
    a.set_dyn(data.clone(), vec![1, 10, 4096]);
    let b = a * a;
    b.mark();

    cx.compile(<(MetalFp32Compiler, GenericCompiler)>::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec::<Rank3<1, 10, 4096>>(
        data,
        (
            dfdx::prelude::Const::<1>,
            dfdx::prelude::Const::<10>,
            dfdx::prelude::Const::<4096>,
        ),
    );
    let d_b = d_a.clone() * d_a;

    assert_close(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
}

#[test]
fn test_mul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a * b;
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a * d_b;

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_mul2() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(
        crate::prelude::Const<1>,
        crate::prelude::Const<1>,
        Dyn<'a'>,
        Dyn<'a'>,
    )>("Input");
    a.set_dyn(vec![82.4, 783.0, 99.6, 974.5], vec![1, 1, 2, 2]);
    let b = cx.new_tensor::<R0>("Input");
    b.set(vec![0.57735026]);
    let c = a * b.expand();
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[[82.4, 783.0], [99.6, 974.5]]]]);
    let d_b = d_dev.tensor(0.57735026);
    let d_c = d_a * d_b.broadcast::<_, dfdx::shapes::Axes4<0, 1, 2, 3>>();

    assert_close(&c.dyn_data(&cx.dyn_map), &d_c.as_vec());
}

#[test]
fn test_div() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a / b;
    c.mark();

    cx.compile(MetalFp32Compiler::default());
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
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a.max(b);
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]);
    let d_b = d_dev.tensor([1., 2., 3.]);
    let d_c = d_a.maximum(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_mod() {
    let mut cx = Graph::new();
    let a_data = random_vec(3);
    let b_data = random_vec(3);
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(a_data.clone());
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(b_data.clone());
    let c = a % b;
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    // No dfdx equivalent

    assert_close(
        &c.data(),
        &a_data
            .into_iter()
            .zip(b_data)
            .map(|(a, b)| a % b)
            .collect_vec(),
    );
}

// Reduction op tests

#[test]
fn test_sum_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(4096);
    let a = cx.new_tensor::<R3<1, 1, 4096>>("Input");
    a.set(data.clone());
    let b = a.sum_reduce::<_, LAxis<1>>();
    let c = a.sum_reduce::<_, LAxis<0>>();
    let d = a.sum_reduce::<_, LAxis<2>>();
    b.mark();
    c.mark();
    d.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<1>, DConst::<4096>));
    let d_b = d_a.clone().sum::<_, DAxis<1>>();
    let d_c = d_a.clone().sum::<_, DAxis<0>>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_max_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.new_tensor::<R3<2, 2, 3>>("Input");
    a.set(data.clone());
    let b = a.max_reduce::<_, LAxis<1>>();
    let c = a.max_reduce::<_, LAxis<0>>();
    let d = a.max_reduce::<_, LAxis<2>>();
    b.mark();
    c.mark();
    d.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<2>, DConst::<2>, DConst::<3>));
    let d_b = d_a.clone().max::<_, DAxis<1>>();
    let d_c = d_a.clone().max::<_, DAxis<0>>();
    let d_d = d_a.max::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_mean_reduce() {
    let data = random_vec(40960);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<1, 10, 4096>>("Input");
    a.set(data.clone());
    let b = a.mean_reduce::<_, LAxis<2>>();
    b.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<4096>));
    let d_b = d_a.mean::<_, DAxis<2>>();
    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a_data = random_vec(2 * 4096);
    let b_data = random_vec(4096 * 4);
    let a = cx.new_tensor::<R2<2, 4096>>("Input");
    a.set(a_data.clone());
    let b = cx.new_tensor::<R2<4096, 4>>("Input");
    b.set(b_data.clone());
    let c = a.matmul(b);
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<4096>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<4096>, DConst::<4>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<2, 2, 3>>("Input");
    a.set(vec![1., 2., 3., 1., 2., 1., 1., 2., 3., 1., 2., 1.]);
    let b = cx.new_tensor::<R2<3, 4>>("Input");
    b.set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);
    let c = a.matmul(b);
    c.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 1.]], [[1., 2., 3.], [1., 2., 1.]]]);
    let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 1., 2.], [-1., -2., 1., 2.]]);
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_matmul_transpose() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R2<2, 3>>("Input");
    a.set(vec![1., 2., 3., 1., 2., 1.]);
    let b = cx.new_tensor::<R2<4, 3>>("Input");
    b.set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);
    let a_t = cx.new_tensor::<R2<3, 2>>("Input");
    a_t.set(vec![1., 2., 3., 1., 2., 1.]);
    let b_t = cx.new_tensor::<R2<3, 4>>("Input");
    b_t.set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);

    let a_b = a.matmul(b.permute());
    let a_b_t = a.matmul(b_t);
    let a_t_b = a_t.permute::<R2<2, 3>, _>().matmul(b.permute());
    let a_t_b_t = a_t.permute::<R2<2, 3>, _>().matmul(b_t);
    a_b.mark();
    a_b_t.mark();
    a_t_b.mark();
    a_t_b_t.mark();

    cx.compile(MetalFp32Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 1.]]);
    let d_b = d_dev.tensor([[1., 2., 3.], [1., 1., 2.], [1., 2., -1.], [-2., 1., 2.]]);
    let d_a_t = d_dev.tensor([[1., 2.], [3., 1.], [2., 1.]]);
    let d_b_t = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 1., 2.], [-1., -2., 1., 2.]]);
    let d_a_b = d_a.clone().matmul(d_b.clone().permute());
    let d_a_b_t = d_a.matmul(d_b_t.clone());
    let d_a_t_b = d_a_t
        .clone()
        .permute::<Rank2<2, 3>, _>()
        .matmul(d_b.permute());
    let d_a_t_b_t = d_a_t.permute::<Rank2<2, 3>, _>().matmul(d_b_t);

    assert_close(&a_b.data(), &d_a_b.as_vec());
    assert_close(&a_b_t.data(), &d_a_b_t.as_vec());
    assert_close(&a_t_b.data(), &d_a_t_b.as_vec());
    assert_close(&a_t_b_t.data(), &d_a_t_b_t.as_vec());
}

#[test]
fn test_relu_and_linear() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let batch = cx.new_tensor::<R2<2, 3>>("Input");
    let a = cx.new_tensor::<R1<3>>("Input");

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

    let unoptimized_b = b.data();
    let unoptimized_batch_out = batch_out.data();
    cx.compile(<(MetalFp32Compiler, GenericCompiler)>::default());
    cx.execute();

    assert_close(&unoptimized_b, &b.data());
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
    let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
    let out = model.forward(a);

    assert_close(&unoptimized_b, &out.as_vec());
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

    let a = cx.new_tensor::<(Dyn<'b'>, Dyn<'a'>, crate::prelude::Const<3>)>("Input");
    let b = model.forward(a);

    a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], vec![1, 2, 3]);
    b.mark();

    cx.compile(<(MetalFp32Compiler, GenericCompiler)>::default());
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

    assert_close(&b.dyn_data(&cx.dyn_map), &d_b.as_vec());
}
