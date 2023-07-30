use std::ops::{Add, Mul};

use crate::prelude::*;

/// Rectified Linear Unit activation function
pub struct ReLU;

impl InitModule for ReLU {
    fn initialize(_: &mut Graph) -> Self {
        Self
    }
}

impl SerializeModule for ReLU {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: Shape> Module<GraphTensor<S>> for ReLU {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.relu()
    }
}

/// Sigmoid activation function
pub struct Sigmoid;

impl InitModule for Sigmoid {
    fn initialize(_: &mut Graph) -> Self {
        Self
    }
}

impl SerializeModule for Sigmoid {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Sigmoid {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.sigmoid()
    }
}

/// Swish activation function
pub struct Swish;

impl InitModule for Swish {
    fn initialize(_: &mut Graph) -> Self {
        Self
    }
}

impl SerializeModule for Swish {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Swish {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.swish()
    }
}

/// Tanh activation function
pub struct Tanh;

impl InitModule for Tanh {
    fn initialize(_: &mut Graph) -> Self {
        Self
    }
}

impl SerializeModule for Tanh {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Tanh {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.tanh()
    }
}

/// RMSNorm activation function
pub struct RMSNorm<const DIM: usize> {
    weight: GraphTensor<R1<DIM>>,
}

impl<const DIM: usize> InitModule for RMSNorm<DIM> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.new_tensor(),
        };
        s.weight.set(vec![1.0; DIM]);
        s
    }
}

impl<const DIM: usize> SerializeModule for RMSNorm<DIM> {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
    }
}

impl<const DIM: usize> Module<GraphTensor<R1<DIM>>> for RMSNorm<DIM> {
    type Output = GraphTensor<R1<DIM>>;

    fn forward(&self, input: GraphTensor<R1<DIM>>) -> Self::Output {
        (input * input)
            .mean_reduce::<_, Axis<0>>()
            .add(1e-6)
            .sqrt()
            .recip()
            .expand()
            .mul(input)
            .mul(self.weight)
    }
}

impl<S: Dim, const DIM: usize> Module<GraphTensor<(S, Const<DIM>)>> for RMSNorm<DIM> {
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        (input * input)
            .mean_reduce::<_, Axis<0>>()
            .add(1e-6)
            .sqrt()
            .recip()
            .expand()
            .mul(input)
            .mul(self.weight.expand())
    }
}

impl<B: Dim, S: Dim, const DIM: usize> Module<GraphTensor<(B, S, Const<DIM>)>> for RMSNorm<DIM> {
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        (input * input)
            .mean_reduce::<_, Axis<0>>()
            .add(1e-6)
            .sqrt()
            .recip()
            .expand()
            .mul(input)
            .mul(self.weight.expand())
    }
}

#[cfg(test)]
mod tests {
    use super::ReLU;
    use crate::{
        nn::linear::Linear,
        prelude::{Module, *},
        tests::{assert_close, assert_close_data},
    };
    use dfdx::prelude::{Module as DfdxModule, *};

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

        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
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
        let d_batch = dev.tensor_from_vec(
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let out = model.forward(a);
        let d_batch_out = model.forward(d_batch);

        assert_close_data(&unoptimized_b.real_data().unwrap(), &out.as_vec());
        assert_close_data(
            &unoptimized_batch_out.real_data().unwrap(),
            &d_batch_out.as_vec(),
        );
    }
}
