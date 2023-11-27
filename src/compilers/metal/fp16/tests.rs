use dfdx::prelude::{Module as DfdxModule, *};
use half::f16;
use itertools::Itertools;
use num_traits::Float;
use rand::{rngs::StdRng, SeedableRng};

use super::MetalFp16Compiler;
use crate::{
    nn::{
        activation::{RMSNorm, ReLU},
        linear::Linear,
    },
    prelude::{Module, *},
    tests::assert_close_precision,
};

crate::test_imports!();

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.new_tensor::<R2<3, 4>>("Input");
    a.set(data.clone());
    let b = a.permute::<R2<4, 3>, _>().reshape::<R2<12, 1>>();
    b.retrieve();
    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<3>, DConst::<4>))
        .to_dtype::<f16>();
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
}

#[test]
fn test_log2() {
    let mut cx = Graph::new();
    let data = random_vec(3);
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(data.clone());
    let b = a.log_2();
    b.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    assert_close(
        &b.data(),
        &data
            .into_iter()
            .map(|i| f16::from_f32(i).log2().to_f32())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_exp2() {
    let mut cx = Graph::new();
    let data = random_vec(3);
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(data.clone());
    let b = a.exp_2();
    b.retrieve();

    cx.compile(MetalFp16Compiler::default());
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
    b.retrieve();
    cx.compile(MetalFp16Compiler::default());
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
    b.retrieve();
    cx.compile(MetalFp16Compiler::default());
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
    b.retrieve();
    cx.compile(MetalFp16Compiler::default());
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
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_c = d_a + d_b;

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_sub() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a - b;
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_c = d_a - d_b;

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_square() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(Dyn<'b'>, Dyn<'s'>, crate::prelude::Const<4096>)>("Input");
    let mut rng = rand::thread_rng();
    let data = random_vec_rng(40960, &mut rng);
    a.set_dyn(data.clone(), vec![1, 10, 4096]);
    let b = a * a;
    b.retrieve();

    cx.compile(<(MetalFp16Compiler, GenericCompiler)>::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec::<Rank3<1, 10, 4096>>(
            data,
            (
                dfdx::prelude::Const::<1>,
                dfdx::prelude::Const::<10>,
                dfdx::prelude::Const::<4096>,
            ),
        )
        .to_dtype::<f16>();
    let d_b = d_a.clone() * d_a;

    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
}

#[test]
fn test_mul() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a * b;
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_c = d_a * d_b;

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_mul2() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<(LConst<1>, LConst<1>, Dyn<'a'>, Dyn<'a'>)>("Input");
    a.set_dyn(vec![82.4, 783.0, 99.6, 974.5], vec![1, 1, 2, 2]);
    let b = cx.new_tensor::<R0>("Input");
    b.set(vec![0.57735026]);
    let c = a * b.expand();
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor([[[[82.4, 783.0], [99.6, 974.5]]]])
        .to_dtype::<f16>();
    let d_b = d_dev.tensor(0.57735026).to_dtype::<f16>();
    let d_c = d_a * d_b.broadcast::<_, dfdx::shapes::Axes4<0, 1, 2, 3>>();

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_div() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a / b;
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_c = d_a / d_b;

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_max() {
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<3>>("Input");
    a.set(vec![1., 2., 3.]);
    let b = cx.new_tensor::<R1<3>>("Input");
    b.set(vec![1., 2., 3.]);
    let c = a.max(b);
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<f16>();
    let d_c = d_a.maximum(d_b);

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
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
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
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
    let data = random_vec(40960);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<1, 10, 4096>>("Input");
    a.set(data.clone());
    let b = a.sum_reduce::<_, LAxis<2>>();
    b.retrieve();
    let c = a.sum_reduce::<_, LAxis<1>>();
    c.retrieve();
    let d = a.sum_reduce::<_, LAxis<0>>();
    d.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<4096>))
        .to_dtype::<f16>();
    let d_b = d_a.clone().sum::<_, DAxis<2>>();
    let d_c = d_a.clone().sum::<_, DAxis<1>>();
    let d_d = d_a.sum::<_, DAxis<0>>();
    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
    assert_close(&d.data(), &d_d.to_dtype::<f32>().as_vec());
}

#[test]
fn test_sum_reduce2() {
    let mut cx = Graph::new();
    let data = random_vec(32 * 10 * 10 * 128);
    let a = cx.new_tensor::<R5<1, 32, 10, 10, 128>>("Input");
    a.set(data.clone());
    let d = a.sum_reduce::<_, LAxis<2>>();
    d.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(
            data,
            (
                DConst::<1>,
                DConst::<32>,
                DConst::<10>,
                DConst::<10>,
                DConst::<128>,
            ),
        )
        .to_dtype::<f16>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_exact(&d.data(), &d_d.to_dtype::<f32>().as_vec());
}

#[test]
fn test_max_reduce() {
    let data = random_vec(40960);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<1, 10, 4096>>("Input");
    a.set(data.clone());
    let b = a.max_reduce::<_, LAxis<2>>();
    b.retrieve();
    let c = a.max_reduce::<_, LAxis<1>>();
    c.retrieve();
    let d = a.max_reduce::<_, LAxis<0>>();
    d.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<4096>))
        .to_dtype::<f16>();
    let d_b = d_a.clone().max::<_, DAxis<2>>();
    let d_c = d_a.clone().max::<_, DAxis<1>>();
    let d_d = d_a.max::<_, DAxis<0>>();
    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
    assert_close(&d.data(), &d_d.to_dtype::<f32>().as_vec());
}

#[test]
fn test_mean_reduce() {
    let data = random_vec(40960);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R3<1, 10, 4096>>("Input");
    a.set(data.clone());
    let b = a.mean_reduce::<_, LAxis<2>>();
    b.retrieve();
    let c = a.mean_reduce::<_, LAxis<1>>();
    c.retrieve();
    let d = a.mean_reduce::<_, LAxis<0>>();
    d.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<4096>))
        .to_dtype::<f16>();
    let d_b = d_a.clone().mean::<_, DAxis<2>>();
    let d_c = d_a.clone().mean::<_, DAxis<1>>();
    let d_d = d_a.mean::<_, DAxis<0>>();
    assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
    assert_close(&d.data(), &d_d.to_dtype::<f32>().as_vec());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let a_data = random_vec_rng(2 * 4096, &mut rng);
    let b_data = random_vec_rng(4 * 4096, &mut rng);
    let a = cx.new_tensor::<R2<2, 4096>>("Input").set(a_data.clone());
    let b = cx.new_tensor::<R2<4096, 4>>("Input").set(b_data.clone());
    let c = a.matmul(b).retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(a_data, (DConst::<2>, DConst::<4096>))
        .to_dtype::<f16>();
    let d_b = d_dev
        .tensor_from_vec(b_data, (DConst::<4096>, DConst::<4>))
        .to_dtype::<f16>();
    let d_c = d_a.matmul(d_b);

    assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
}

#[test]
fn test_attn_matmul() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let a_data = random_vec_rng(32 * 11 * 128, &mut rng);
    let b_data = random_vec_rng(32 * 11 * 128, &mut rng);
    let a = cx.new_tensor::<R4<1, 32, 11, 128>>("Input");
    a.set(a_data.clone());
    a.keep();
    let b = cx.new_tensor::<R4<1, 32, 128, 11>>("Input");
    b.set(b_data.clone());
    b.keep();
    let c = a.matmul(b);
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(
            a_data,
            (DConst::<1>, DConst::<32>, DConst::<11>, DConst::<128>),
        )
        .to_dtype::<f16>();
    let d_b = d_dev
        .tensor_from_vec(
            b_data,
            (DConst::<1>, DConst::<32>, DConst::<128>, DConst::<11>),
        )
        .to_dtype::<f16>();
    let d_c = d_a.matmul(d_b);
    assert_exact(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let a_data = random_vec_rng(2 * 20 * 128, &mut rng);
    let b_data = random_vec_rng(128 * 128, &mut rng);
    let a = cx.new_tensor::<R3<2, 20, 128>>("Input").set(a_data.clone());
    let b = cx.new_tensor::<R2<128, 128>>("Input").set(b_data.clone());
    let c = a.matmul(b).retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(a_data, (DConst::<2>, DConst::<20>, DConst::<128>))
        .to_dtype::<f16>();
    let d_b = d_dev
        .tensor_from_vec(b_data, (DConst::<128>, DConst::<128>))
        .to_dtype::<f16>();
    let d_c = d_a.matmul(d_b);

    assert_exact(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_batch_matmul_transpose() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let a_data = random_vec_rng(2 * 10 * 128, &mut rng);
    let b_data = random_vec_rng(128 * 128, &mut rng);
    let a = cx.new_tensor::<R3<2, 10, 128>>("Input").set(a_data.clone());
    let b = cx.new_tensor::<R2<128, 128>>("Input").set(b_data.clone());
    let c = a
        // .permute::<_, LAxes3<0, 2, 1>>()
        .matmul(b.permute::<_, LAxes2<1, 0>>())
        .retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(a_data, (DConst::<2>, DConst::<10>, DConst::<128>))
        .to_dtype::<f16>();
    let d_b = d_dev
        .tensor_from_vec(b_data, (DConst::<128>, DConst::<128>))
        .to_dtype::<f16>();
    let d_c = d_a
        // .permute::<_, DAxes3<0, 2, 1>>()
        .matmul(d_b.permute::<_, DAxes2<1, 0>>());

    assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_matmul_transpose() {
    let mut cx = Graph::new();
    let a = cx
        .new_tensor::<R2<2, 3>>("Input")
        .set(vec![1., 2., 3., 1., 2., 1.]);
    let b = cx
        .new_tensor::<R2<4, 3>>("Input")
        .set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);
    let a_t = cx
        .new_tensor::<R2<3, 2>>("Input")
        .set(vec![1., 2., 3., 1., 2., 1.]);
    let b_t = cx
        .new_tensor::<R2<3, 4>>("Input")
        .set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);

    let a_b = a.matmul(b.permute()).retrieve();
    let a_b_t = a.matmul(b_t).retrieve();
    let a_t_b = a_t.permute::<R2<2, 3>, _>().matmul(b.permute()).retrieve();
    let a_t_b_t = a_t.permute::<R2<2, 3>, _>().matmul(b_t).retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 1.]]).to_dtype::<f16>();
    let d_b = d_dev
        .tensor([[1., 2., 3.], [1., 1., 2.], [1., 2., -1.], [-2., 1., 2.]])
        .to_dtype::<f16>();
    let d_a_t = d_dev
        .tensor([[1., 2.], [3., 1.], [2., 1.]])
        .to_dtype::<f16>();
    let d_b_t = d_dev
        .tensor([[1., 2., 3., 1.], [1., 2., 1., 2.], [-1., -2., 1., 2.]])
        .to_dtype::<f16>();
    let d_a_b = d_a.clone().matmul(d_b.clone().permute());
    let d_a_b_t = d_a.matmul(d_b_t.clone());
    let d_a_t_b = d_a_t
        .clone()
        .permute::<Rank2<2, 3>, _>()
        .matmul(d_b.permute());
    let d_a_t_b_t = d_a_t.permute::<Rank2<2, 3>, _>().matmul(d_b_t);

    assert_close(&a_b.data(), &d_a_b.to_dtype::<f32>().as_vec());
    assert_close(&a_b_t.data(), &d_a_b_t.to_dtype::<f32>().as_vec());
    assert_close(&a_t_b.data(), &d_a_t_b.to_dtype::<f32>().as_vec());
    assert_close(&a_t_b_t.data(), &d_a_t_b_t.to_dtype::<f32>().as_vec());
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
    b.retrieve();
    batch_out.retrieve();
    cx.execute();

    let unoptimized_b = b.data();
    let unoptimized_batch_out = batch_out.data();
    cx.compile(<(GenericCompiler, MetalFp16Compiler)>::default());
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
fn test_rms_norm() {
    // Test single and batch, unoptimized and optimized
    let inp_data = random_vec(3 * 4);
    let weight_data = random_vec(4);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R2<3, 4>>("Input");

    let model = RMSNorm::<4>::initialize(&mut cx);
    model.weight.set(weight_data.clone());
    let b = model.forward(a);
    a.set(inp_data.clone());
    b.retrieve();

    cx.compile(<(MetalFp16Compiler, GenericCompiler)>::default());
    cx.execute();

    // Test against dfdx
    let dev = Cpu::default();
    let weight = dev
        .tensor_from_vec(weight_data, (DConst::<4>,))
        .to_dtype::<f16>();
    let a = dev
        .tensor_from_vec(inp_data, (DConst::<3>, DConst::<4>))
        .to_dtype::<f16>()
        .to_dtype::<f32>();
    let var_f32 = a.clone().square().mean::<_, DAxis<1>>();
    let inv_std_f32 = (var_f32 + 1e-6).sqrt().recip();
    let x_f32 = inv_std_f32.broadcast() * a;
    let out = weight.broadcast() * x_f32.to_dtype::<f16>();

    assert_exact(
        &b.data().into_iter().map(f16::from_f32).collect::<Vec<_>>(),
        &out.as_vec(),
    );
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

    let a = cx
        .new_tensor::<(Dyn<'b'>, Dyn<'a'>, crate::prelude::Const<3>)>("Input")
        .set_dyn(vec![-1., 2., 3., 3., 3., -1.], vec![1, 2, 3]);
    let b = model.forward(a).retrieve();

    cx.compile(<(MetalFp16Compiler, GenericCompiler)>::default());
    cx.execute();

    let d_dev = Cpu::default();
    let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
        d_dev.build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
    d_model.self_attn.w_k.bias.copy_from(&[0.; 3]);
    d_model.self_attn.w_v.bias.copy_from(&[0.; 3]);
    d_model.self_attn.w_q.bias.copy_from(&[0.; 3]);
    d_model.self_attn.w_o.bias.copy_from(&[0.; 3]);
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

    // Annoyingly dfdx transformer encoder outputs 0s in fp16 mode, so we need to use the fp32 mode. Result ends up being close enough
    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_common_buffer() {
    let data = random_vec(32);
    let mut cx = Graph::new();
    let a = cx.new_tensor::<R1<32>>("Input");
    a.set(data.clone());
    let a1 = cx.new_tensor::<R1<32>>("Input1");
    a1.set(data.clone());
    let exped = a * a1;
    let b = exped.log_2();
    let c = exped.sin();
    b.retrieve();
    c.retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();
}

#[test]
fn test_embedding() {
    let mut cx = Graph::new();
    let batch = cx
        .new_tensor::<R2<2, 3>>("BatchInput")
        .set(vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0])
        .keep();
    let a = cx
        .new_tensor::<R1<3>>("Input")
        .set(vec![1.0, 0.0, 1.0])
        .keep();

    let model: crate::nn::embedding::Embedding<3, 4> = InitModule::initialize(&mut cx);
    model
        .weight
        .set(vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.]);
    let b = model.forward(a).retrieve();
    let batch_out = model.forward(batch).retrieve();

    cx.compile(MetalFp16Compiler::default());
    cx.execute();

    let d_dev = Cpu::default();
    let mut d_model = <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
    d_model.weight = d_dev.tensor_from_vec(
        vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.],
        (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
    );
    let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (dfdx::shapes::Const::<3>,));
    let d_batch = d_dev.tensor_from_vec(
        vec![1, 0, 2, 1, 0, 1],
        (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
    );

    let d_b = d_model.forward(d_a);
    let d_batch_out = d_model.forward(d_batch);

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&batch_out.data(), &d_batch_out.as_vec());
}
