use crate::{
    nn::{activation::ReLU, linear::Linear},
    prelude::*,
};

use super::attention::MultiHeadSelfAttention;

/// A transformer decoder as layed out in *Attention Is All You Need*.
pub struct TransformerDecoder<
    const DIM: usize,
    const FF: usize,
    const HEADS: usize,
    const LAYERS: usize,
> {
    layers: Vec<TransformerDecoderBlock<DIM, FF, HEADS>>,
}

// Single
impl<const DIM: usize, const FF: usize, const HEADS: usize, const LAYERS: usize, S: Dim>
    Module<(GraphTensor<(S, Const<DIM>)>, GraphTensor<(S, Const<DIM>)>)>
    for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(
        &self,
        (mut input, from_enc): (GraphTensor<(S, Const<DIM>)>, GraphTensor<(S, Const<DIM>)>),
    ) -> Self::Output {
        for layer in &self.layers {
            input = layer.forward((input, from_enc));
        }
        input
    }
}

// Batched
impl<
        const DIM: usize,
        const FF: usize,
        const HEADS: usize,
        const LAYERS: usize,
        B: Dim,
        S: Dim,
    >
    Module<(
        GraphTensor<(B, S, Const<DIM>)>,
        GraphTensor<(B, S, Const<DIM>)>,
    )> for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(
        &self,
        (mut input, from_enc): (
            GraphTensor<(B, S, Const<DIM>)>,
            GraphTensor<(B, S, Const<DIM>)>,
        ),
    ) -> Self::Output {
        for layer in &self.layers {
            input = layer.forward((input, from_enc));
        }
        input
    }
}

/// A single transformer decoder block
pub struct TransformerDecoderBlock<const DIM: usize, const FF: usize, const HEADS: usize> {
    pub(crate) self_attention: MultiHeadSelfAttention<DIM, DIM, DIM, HEADS>,
    pub(crate) cross_attention: MultiHeadSelfAttention<DIM, DIM, DIM, HEADS>,
    pub(crate) ff: (Linear<DIM, FF>, ReLU, Linear<FF, DIM>),
}

impl<const DIM: usize, const FF: usize, const HEADS: usize> InitModule
    for TransformerDecoderBlock<DIM, FF, HEADS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            cross_attention: InitModule::initialize(cx),
            self_attention: InitModule::initialize(cx),
            ff: InitModule::initialize(cx),
        }
    }
}

// Single
impl<const DIM: usize, const FF: usize, const HEADS: usize, S: Dim>
    Module<(GraphTensor<(S, Const<DIM>)>, GraphTensor<(S, Const<DIM>)>)>
    for TransformerDecoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(
        &self,
        (input, from_enc): (GraphTensor<(S, Const<DIM>)>, GraphTensor<(S, Const<DIM>)>),
    ) -> Self::Output {
        // Pass to batched forward
        <Self as Module<(
            GraphTensor<(Const<1>, S, Const<DIM>)>,
            GraphTensor<(Const<1>, S, Const<DIM>)>,
        )>>::forward(self, (input.expand(), from_enc.expand()))
        .max_reduce::<_, Axis<0>>()
    }
}

// Batched
impl<const DIM: usize, const FF: usize, const HEADS: usize, S: Dim, B: Dim>
    Module<(
        GraphTensor<(B, S, Const<DIM>)>,
        GraphTensor<(B, S, Const<DIM>)>,
    )> for TransformerDecoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(
        &self,
        (input, from_enc): (
            GraphTensor<(B, S, Const<DIM>)>,
            GraphTensor<(B, S, Const<DIM>)>,
        ),
    ) -> Self::Output {
        let x = self.self_attention.forward(input);
        let x = (x + input).layer_norm::<2>();
        let y = self.cross_attention.forward((from_enc, x, from_enc));
        let x = (y + x).layer_norm::<2>();
        let y = self.ff.forward(x);
        (y + x).layer_norm::<2>()
    }
}
