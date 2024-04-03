use crate::{Linear, ReLU};
use luminal::prelude::*;

use super::attention::MultiHeadSelfAttention;

/// A transformer decoder as layed out in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
pub struct TransformerDecoder<
    const DIM: usize,
    const FF: usize,
    const HEADS: usize,
    const LAYERS: usize,
> {
    pub layers: Vec<TransformerDecoderBlock<DIM, FF, HEADS>>,
}

impl<const DIM: usize, const FF: usize, const HEADS: usize, const LAYERS: usize> InitModule
    for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            layers: (0..LAYERS).map(|_| InitModule::initialize(cx)).collect(),
        }
    }
}

impl<const DIM: usize, const FF: usize, const HEADS: usize, const LAYERS: usize> SerializeModule
    for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    fn serialize(&self, s: &mut Serializer) {
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("layer{i}"), l);
        }
    }
}

// Single
impl<
        const DIM: usize,
        const FF: usize,
        const HEADS: usize,
        const LAYERS: usize,
        S1: Dimension,
        S2: Dimension,
    > Module<(GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>)>
    for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    type Output = GraphTensor<(S1, Const<DIM>)>;

    fn forward(
        &self,
        (input, from_enc): (GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>),
    ) -> Self::Output {
        <Self as Module<(
            GraphTensor<(Const<1>, S1, Const<DIM>)>,
            GraphTensor<(Const<1>, S2, Const<DIM>)>,
        )>>::forward(self, (input.expand(), from_enc.expand()))
        .max_reduce()
    }
}

// Batched
impl<
        const DIM: usize,
        const FF: usize,
        const HEADS: usize,
        const LAYERS: usize,
        B: Dimension,
        S1: Dimension,
        S2: Dimension,
    >
    Module<(
        GraphTensor<(B, S1, Const<DIM>)>,
        GraphTensor<(B, S2, Const<DIM>)>,
    )> for TransformerDecoder<DIM, FF, HEADS, LAYERS>
{
    type Output = GraphTensor<(B, S1, Const<DIM>)>;

    fn forward(
        &self,
        (mut input, from_enc): (
            GraphTensor<(B, S1, Const<DIM>)>,
            GraphTensor<(B, S2, Const<DIM>)>,
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

impl<const DIM: usize, const FF: usize, const HEADS: usize> SerializeModule
    for TransformerDecoderBlock<DIM, FF, HEADS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.self_attention);
        s.module("cross_attn", &self.cross_attention);
        s.module("ff", &self.ff);
    }
}

// Single
impl<const DIM: usize, const FF: usize, const HEADS: usize, S1: Dimension, S2: Dimension>
    Module<(GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>)>
    for TransformerDecoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(S1, Const<DIM>)>;

    fn forward(
        &self,
        (input, from_enc): (GraphTensor<(S1, Const<DIM>)>, GraphTensor<(S2, Const<DIM>)>),
    ) -> Self::Output {
        // Pass to batched forward
        <Self as Module<(
            GraphTensor<(Const<1>, S1, Const<DIM>)>,
            GraphTensor<(Const<1>, S2, Const<DIM>)>,
        )>>::forward(self, (input.expand(), from_enc.expand()))
        .max_reduce()
    }
}

// Batched
impl<
        const DIM: usize,
        const FF: usize,
        const HEADS: usize,
        S1: Dimension,
        S2: Dimension,
        B: Dimension,
    >
    Module<(
        GraphTensor<(B, S1, Const<DIM>)>,
        GraphTensor<(B, S2, Const<DIM>)>,
    )> for TransformerDecoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(B, S1, Const<DIM>)>;

    fn forward(
        &self,
        (x, from_enc): (
            GraphTensor<(B, S1, Const<DIM>)>,
            GraphTensor<(B, S2, Const<DIM>)>,
        ),
    ) -> Self::Output {
        let y = self.self_attention.forward(x);
        let x = (y + x).layer_norm::<Axis<2>, _>(1e-5);
        let y = self.cross_attention.forward((from_enc, x, from_enc));
        let x = (y + x).layer_norm::<Axis<2>, _>(1e-5);
        let y = self.ff.forward(x);
        (y + x).layer_norm::<Axis<2>, _>(1e-5)
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::{DeviceBuildExt, Module as DfdxModule},
        tensor::{Cpu, TensorFromVec},
        tensor_ops::PermuteTo,
    };

    use luminal::{
        prelude::{Module, *},
        tests::assert_close,
    };

    use super::TransformerDecoderBlock;
    #[test]
    fn test_transformer_decoder_block() {
        let mut cx = Graph::new();
        let model: TransformerDecoderBlock<3, 4, 1> = InitModule::initialize(&mut cx);
        model
            .self_attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .self_attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .self_attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .self_attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .cross_attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .cross_attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .cross_attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .cross_attention
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

        let a = cx.tensor::<(Dyn<'d'>, Const<3>)>();
        let e = cx.tensor::<(Dyn<'e'>, Const<3>)>();
        let b = model.forward((a, e));

        a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], &[2, 3]);
        e.set_dyn(vec![-1., 2., 3., 3., 3., -1., -1., 2., 3.], &[3, 3]);
        b.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::TransformerDecoderBlock<3, 1, 4, f32, Cpu> =
            d_dev
                .build_module::<dfdx::nn::modules::builders::TransformerDecoderBlock<3, 1, 4>, f32>(
                );
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
        d_model.mh_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.mh_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.mh_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.mh_attn.w_o.bias.copy_from(&[0., 0., 0.]);
        d_model.mh_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.mh_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.mh_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.mh_attn.w_v.weight = d_dev
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
        d_model.ff.0 .0.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
        d_model.ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm3.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm3.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.epsilon = 1e-5;
        d_model.norm2.epsilon = 1e-5;
        d_model.norm3.epsilon = 1e-5;
        let d_a = d_dev.tensor_from_vec(
            vec![-1., 2., 3., 3., 3., -1.],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_e = d_dev.tensor_from_vec(
            vec![-1., 2., 3., 3., 3., -1., -1., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward((d_a, d_e));

        assert_close(&b.data(), &d_b.as_vec());
    }
}
