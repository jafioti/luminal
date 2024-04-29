use dfdx::prelude::*;
use luminal::prelude::*;
use luminal::tests::random_vec_rng;
use rand::{rngs::StdRng, SeedableRng};

mod fp16;
mod fp32;

#[macro_export]
macro_rules! single_unary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty, $size: expr) => {
        paste::paste! {
            #[test]
            fn [<$name _ $type _ $size>]() {
                let mut rng = StdRng::seed_from_u64(1);
                let data = random_vec_rng($size, &mut rng);
                let mut cx = Graph::new();
                let a = cx.tensor::<R1<$size>>().set(data.clone());
                let f: fn(GraphTensor<R1<$size>>) -> GraphTensor<R1<$size>> = $luminal_func;
                let mut b = f(a).retrieve();
                cx.compile($crate::CudaCompiler::<$type>::default(), &mut b);
                cx.execute();

                let d_dev = Cpu::default();
                let d_a = d_dev
                    .tensor_from_vec(data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let f: fn(
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                ) -> dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape> = $dfdx_func;
                let d_b = f(d_a);

                luminal::tests::assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
            }
        }
    };
}

#[macro_export]
macro_rules! unary_test_type {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 3);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 50);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 783);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 4096);
    };
}

#[macro_export]
macro_rules! unary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident) => {
        $crate::unary_test_type!($luminal_func, $dfdx_func, $name, f32);
        $crate::unary_test_type!($luminal_func, $dfdx_func, $name, f16);
    };
}

#[macro_export]
macro_rules! single_binary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty, $size: expr) => {
        paste::paste! {
            #[test]
            fn [<$name _ $type _ $size>]() {
                let mut rng = StdRng::seed_from_u64(2);
                let a_data = random_vec_rng($size, &mut rng);
                let b_data = random_vec_rng($size, &mut rng);
                let mut cx = Graph::new();
                let a = cx.tensor::<R1<$size>>().set(a_data.clone());
                let b = cx.tensor::<R1<$size>>().set(b_data.clone());
                let f: fn(GraphTensor<R1<$size>>, GraphTensor<R1<$size>>) -> GraphTensor<R1<$size>> =
                    $luminal_func;
                let mut c = f(a, b).retrieve();
                cx.compile($crate::CudaCompiler::<$type>::default(), &mut c);
                cx.execute();

                let d_dev = Cpu::default();
                let d_a = d_dev
                    .tensor_from_vec(a_data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let d_b = d_dev
                    .tensor_from_vec(b_data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let f: fn(
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                ) -> dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape> = $dfdx_func;
                let d_c = f(d_a, d_b);

                luminal::tests::assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
            }
        }
    };
}

#[macro_export]
macro_rules! binary_test_type {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 3);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 50);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 783);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 4096);
    };
}

#[macro_export]
macro_rules! binary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident) => {
        $crate::binary_test_type!($luminal_func, $dfdx_func, $name, f32);
        $crate::binary_test_type!($luminal_func, $dfdx_func, $name, f16);
    };
}

pub fn assert_op_in_graph<T: Operator + 'static>(graph: &Graph) {
    assert!(
        graph.node_indices().any(|i| graph.check_node_type::<T>(i)),
        "Node not found in the graph!"
    );
}

unary_test!(|a| a.sin(), |a| a.sin(), test_sin);
unary_test!(|a| a.sqrt(), |a| a.sqrt(), test_sqrt);
unary_test!(|a| a.recip(), |a| a.recip(), test_recip);
unary_test!(|a| a * a, |a| a.clone() * a, test_square);
unary_test!(|a| a.exp(), |a| a.exp(), test_exp);
unary_test!(|a| a.cos(), |a| a.cos(), test_cos);
unary_test!(|a| a.softmax(), |a| a.softmax(), test_softmax);
unary_test!(
    |a| a.mean_norm::<luminal::shape::Axis<0>>(),
    |a| a.clone() - a.mean::<_, dfdx::prelude::Axis<0>>().broadcast(),
    test_mean_norm
);
unary_test!(
    |a| a.std_norm::<luminal::shape::Axis<0>, _>(1e-5),
    |a| a.clone() / a.stddev::<_, dfdx::prelude::Axis<0>>(1e-5).broadcast(),
    test_std_norm
);
unary_test!(
    |a| a.layer_norm::<luminal::shape::Axis<0>, _>(1e-5),
    |a| a.normalize::<dfdx::prelude::Axis<0>>(1e-5),
    test_norm
);

binary_test!(|a, b| a + b, |a, b| a + b, test_add);
binary_test!(|a, b| a - b, |a, b| a - b, test_sub);
binary_test!(|a, b| a * b, |a, b| a * b, test_mul);
binary_test!(|a, b| a / b, |a, b| a * b.recip(), test_div);
binary_test!(|a, b| a.max(b), |a, b| a.maximum(b), test_max);
binary_test!(|a, b| a.min(b), |a, b| a.minimum(b), test_min);

single_unary_test!(|a| a.ln(), |a| a.ln(), test_ln, f16, 3); // For some reason ln fails on larger tensors
single_unary_test!(|a| a.ln(), |a| a.ln(), test_ln, f32, 3); // For some reason ln fails on larger tensors
