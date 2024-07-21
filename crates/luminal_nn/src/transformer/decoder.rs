use crate::{Linear, ReLU};
use luminal::prelude::*;

use super::attention::MultiHeadSelfAttention;

/// A transformer decoder as layed out in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderBlock>,
}

impl TransformerDecoder {
    pub fn new(dim: usize, ff: usize, heads: usize, layers: usize, cx: &mut Graph) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| TransformerDecoderBlock::new(dim, ff, heads, cx))
                .collect(),
        }
    }
}

impl SerializeModule for TransformerDecoder {
    fn serialize(&self, s: &mut Serializer) {
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("layer{i}"), l);
        }
    }
}

impl Module<(GraphTensor, GraphTensor)> for TransformerDecoder {
    type Output = GraphTensor;

    fn forward(&self, (mut input, from_enc): (GraphTensor, GraphTensor)) -> Self::Output {
        for layer in &self.layers {
            input = layer.forward((input, from_enc));
        }
        input
    }
}

/// A single transformer decoder block
pub struct TransformerDecoderBlock {
    pub self_attention: MultiHeadSelfAttention,
    pub cross_attention: MultiHeadSelfAttention,
    pub ff: (Linear, ReLU, Linear),
}

impl TransformerDecoderBlock {
    pub fn new(dim: usize, ff: usize, heads: usize, cx: &mut Graph) -> Self {
        Self {
            cross_attention: MultiHeadSelfAttention::new(dim, dim, dim, heads, cx),
            self_attention: MultiHeadSelfAttention::new(dim, dim, dim, heads, cx),
            ff: (
                Linear::new(dim, ff, false, cx),
                ReLU,
                Linear::new(ff, dim, false, cx),
            ),
        }
    }
}

impl SerializeModule for TransformerDecoderBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.self_attention);
        s.module("cross_attn", &self.cross_attention);
        s.module("ff", &self.ff);
    }
}

impl Module<(GraphTensor, GraphTensor)> for TransformerDecoderBlock {
    type Output = GraphTensor;

    fn forward(&self, (input, from_enc): (GraphTensor, GraphTensor)) -> Self::Output {
        // Input: batch_dims, seq1, dim
        // From_enc: batch_dims, seq2, dim
        // Flatten to single batch dim
        let seq1 = input.dims()[input.shape.len() - 2];
        let seq2 = from_enc.dims()[from_enc.shape.len() - 2];
        let dim = *input.dims().last().unwrap();
        let n_batches = input
            .dims()
            .into_iter()
            .take(input.shape.len() - 2)
            .product::<Expression>()
            .max(1);
        let inp = input.reshape((n_batches, seq1, dim));
        let fe = from_enc.reshape((n_batches, seq2, dim));
        // Batched forward pass
        let y = self.self_attention.forward(inp);
        let x = (y + inp).layer_norm(2, 1e-5);
        let y = self.cross_attention.forward((fe, x, fe));
        let x = (y + x).layer_norm(2, 1e-5);
        let y = self.ff.forward(x);
        (y + x).layer_norm(2, 1e-5).reshape(input.shape)
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
        let model = TransformerDecoderBlock::new(3, 4, 1, &mut cx);
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

        let a = cx.tensor(('d', 3));
        let e = cx.tensor(('e', 3));
        let b = model.forward((a, e));

        a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], (2, 3));
        e.set_dyn(vec![-1., 2., 3., 3., 3., -1., -1., 2., 3.], (3, 3));
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
