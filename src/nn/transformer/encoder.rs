use crate::{
    nn::{activation::ReLU, linear::Linear, Repeated},
    prelude::*,
};

use super::attention::MultiHeadSelfAttention;

/// A transformer encoder as layed out in *Attention Is All You Need*.
pub type TransformerEncoder<const DIM: usize, const FF: usize, const LAYERS: usize> =
    Repeated<TransformerEncoderBlock<DIM, FF>, LAYERS>;

/// A single transformer encoder block
pub struct TransformerEncoderBlock<const DIM: usize, const FF: usize> {
    pub(crate) attention: MultiHeadSelfAttention<DIM>,
    pub(crate) ff: (Linear<DIM, FF>, ReLU, Linear<FF, DIM>),
}

impl<const DIM: usize, const FF: usize> InitModule for TransformerEncoderBlock<DIM, FF> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            ff: InitModule::initialize(cx),
        }
    }
}

// Single
impl<const DIM: usize, const FF: usize, S: Dim> Module<GraphTensor<(S, Const<DIM>)>>
    for TransformerEncoderBlock<DIM, FF>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        // Pass to batched forward
        <Self as Module<GraphTensor<(Const<1>, S, Const<DIM>)>>>::forward(self, input.expand())
            .max_reduce::<_, Axis<0>>()
    }
}

// Batched
impl<const DIM: usize, const FF: usize, S: Dim, B: Dim> Module<GraphTensor<(B, S, Const<DIM>)>>
    for TransformerEncoderBlock<DIM, FF>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        let x = self.attention.forward(input);
        let x = (x + input).layer_norm::<2>();
        let y = self.ff.forward(x);
        (x + y).layer_norm::<2>()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use super::TransformerEncoder;
    #[test]
    fn test_transformer_encoder() {
        let mut cx = Graph::new();
        let model: TransformerEncoder<3, 4, 2> = InitModule::initialize(&mut cx);

        let a = cx.new_tensor::<(usize, Const<3>)>();
        let b = model.forward(a);

        a.set_dyn(vec![1., 2., 3., 1., 2., 3.], vec![2, 3]);
        b.mark();

        cx.execute();

        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
        cx.execute();
    }
}
