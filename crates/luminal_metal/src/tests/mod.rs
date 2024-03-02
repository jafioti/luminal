mod fp16;
mod fp32;

#[macro_export]
macro_rules! unary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        #[test]
        fn $name() {
            let mut cx = Graph::new();
            let a = cx.tensor::<R1<3>>().set([1., 2., 3.]);
            let f: fn(GraphTensor<R1<3>>) -> GraphTensor<R1<3>> = $luminal_func;
            let mut b = f(a).retrieve();
            cx.compile(MetalCompiler::<$type>::default(), &mut b);
            cx.execute();

            let d_dev = Cpu::default();
            let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<$type>();
            let f: fn(
                dfdx::prelude::Tensor<Rank1<3>, $type, Cpu, NoneTape>,
            ) -> dfdx::prelude::Tensor<Rank1<3>, $type, Cpu, NoneTape> = $dfdx_func;
            let d_b = f(d_a);

            assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
        }
    };
}

#[macro_export]
macro_rules! binary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        #[test]
        fn $name() {
            let mut cx = Graph::new();
            let a = cx.tensor::<R1<3>>().set(vec![1., 2., 3.]);
            let b = cx.tensor::<R1<3>>().set(vec![1., 2., 3.]);
            let f: fn(GraphTensor<R1<3>>, GraphTensor<R1<3>>) -> GraphTensor<R1<3>> = $luminal_func;
            let mut c = f(a, b).retrieve();
            cx.compile(MetalCompiler::<$type>::default(), &mut c);
            cx.execute();

            let d_dev = Cpu::default();
            let d_a = d_dev.tensor([1., 2., 3.]).to_dtype::<$type>();
            let d_b = d_dev.tensor([1., 2., 3.]).to_dtype::<$type>();
            let f: fn(
                dfdx::prelude::Tensor<Rank1<3>, $type, Cpu, NoneTape>,
                dfdx::prelude::Tensor<Rank1<3>, $type, Cpu, NoneTape>,
            ) -> dfdx::prelude::Tensor<Rank1<3>, $type, Cpu, NoneTape> = $dfdx_func;
            let d_c = f(d_a, d_b);

            assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
        }
    };
}
