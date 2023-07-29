use crate::prelude::*;

pub mod attention;
pub mod decoder;
pub mod encoder;

pub struct Transformer<
    const DIM: usize,
    const FF: usize,
    const ENC_HEADS: usize,
    const DEC_HEADS: usize,
    const ENC_LAYERS: usize,
    const DEC_LAYERS: usize,
> {
    encoder: encoder::TransformerEncoder<DIM, FF, ENC_HEADS, ENC_LAYERS>,
    decoder: decoder::TransformerDecoder<DIM, FF, DEC_HEADS, DEC_LAYERS>,
}

// Single Sequence
impl<
        const DIM: usize,
        const FF: usize,
        const ENC_HEADS: usize,
        const DEC_HEADS: usize,
        const ENC_LAYERS: usize,
        const DEC_LAYERS: usize,
        S1: Dim,
        S2: Dim,
    > Module<(GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>)>
    for Transformer<DIM, FF, ENC_HEADS, DEC_HEADS, ENC_LAYERS, DEC_LAYERS>
{
    type Output = GraphTensor<(S2, Const<DIM>)>;

    fn forward(
        &self,
        (input, target): (GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>),
    ) -> Self::Output {
        let encoded = self.encoder.forward(input);
        self.decoder.forward((target, encoded))
    }
}
