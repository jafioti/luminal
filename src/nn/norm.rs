use std::{marker::PhantomData, ops::Mul};

use crate::prelude::*;

/// A simple layer norm layer. Calls `tensor.layer_norm::<DIM>()`.
#[derive(Default)]
pub struct LayerNorm<Ax: Axes>(PhantomData<Ax>);

impl<Ax: Axes> InitModule for LayerNorm<Ax> {
    fn initialize(_: &mut crate::prelude::Graph) -> Self {
        Self::default()
    }
}

impl<Ax: Axes, S: ConstShape> Module<GraphTensor<S>> for LayerNorm<Ax>
where
    S: ReduceShape<Ax>,
    <S as ReduceShape<Ax>>::Reduced: ConstShape,
{
    type Output = GraphTensor<S>;
    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.layer_norm::<Ax, _>(1e-5)
    }
}

impl<Ax: Axes> SerializeModule for LayerNorm<Ax> {
    fn serialize(&self, _: &mut Serializer) {}
}

/// RMSNorm normalization
pub struct RMSNorm<const DIM: usize> {
    pub weight: GraphTensor<R1<DIM>>,
    pub epsilon: f32,
}

impl<const DIM: usize> InitModule for RMSNorm<DIM> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("RMSNorm Weight").set(vec![1.0; DIM]),
            epsilon: 1e-6,
        }
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
        input.std_norm::<Axis<0>, _>(self.epsilon).mul(self.weight)
    }
}

impl<S: Dimension, const DIM: usize> Module<GraphTensor<(S, Const<DIM>)>> for RMSNorm<DIM> {
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        input
            .std_norm::<Axis<1>, _>(self.epsilon)
            .mul(self.weight.expand())
    }
}

impl<B: Dimension, S: Dimension, const DIM: usize> Module<GraphTensor<(B, S, Const<DIM>)>>
    for RMSNorm<DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        input
            .std_norm::<Axis<2>, _>(self.epsilon)
            .mul(self.weight.expand())
    }
}
