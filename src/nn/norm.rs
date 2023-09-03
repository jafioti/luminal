use crate::prelude::*;

/// A simple layer norm layer. Calls `tensor.layer_norm::<DIM>()`.
pub struct LayerNorm<const DIM: isize>;

impl<const DIM: isize> InitModule for LayerNorm<DIM> {
    fn initialize(_: &mut crate::prelude::Graph) -> Self {
        Self
    }
}

impl<const DIM: isize, S: ConstShape> Module<GraphTensor<S>> for LayerNorm<DIM>
where
    S: ReduceShape<Axis<DIM>>,
    <S as ReduceShape<Axis<DIM>>>::Reduced: ConstShape,
{
    type Output = GraphTensor<S>;
    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.layer_norm::<DIM>()
    }
}

impl<const DIM: isize> SerializeModule for LayerNorm<DIM> {
    fn serialize(&self, _: &mut Serializer) {}
}
