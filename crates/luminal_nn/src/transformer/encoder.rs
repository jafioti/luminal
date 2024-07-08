use crate::{Linear, ReLU};
use luminal::prelude::*;

use super::attention::MultiHeadSelfAttention;

/// A transformer encoder as layed out in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderBlock>,
}

impl TransformerEncoder {
    pub fn new(dim: usize, ff: usize, heads: usize, layers: usize, cx: &mut Graph) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| TransformerEncoderBlock::new(dim, ff, heads, cx))
                .collect(),
        }
    }
}

impl SerializeModule for TransformerEncoder {
    fn serialize(&self, s: &mut Serializer) {
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("layer{i}"), l);
        }
    }
}

impl Module<GraphTensor> for TransformerEncoder {
    type Output = GraphTensor;

    fn forward(&self, mut input: GraphTensor) -> Self::Output {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }
}

/// A single transformer encoder block
pub struct TransformerEncoderBlock {
    pub attention: MultiHeadSelfAttention,
    pub ff: (Linear, ReLU, Linear),
}

impl TransformerEncoderBlock {
    pub fn new(dim: usize, ff: usize, heads: usize, cx: &mut Graph) -> Self {
        Self {
            attention: MultiHeadSelfAttention::new(dim, dim, dim, heads, cx),
            ff: (
                Linear::new(dim, ff, false, cx),
                ReLU,
                Linear::new(ff, dim, false, cx),
            ),
        }
    }
}

impl SerializeModule for TransformerEncoderBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("ff", &self.ff);
    }
}

// Batched
impl Module<GraphTensor> for TransformerEncoderBlock {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        // Input: batch_dims, sequence, dim
        // Reshape to 1 batch dim, sequence, dim
        let n_batches = input
            .shape()
            .into_iter()
            .take(input.shape.len() - 2)
            .product::<BigExpression>()
            .max(1);
        let sequence = input.shape()[input.shape.len() - 2].small();
        let dim = input.shape()[input.shape.len() - 1].small();
        let x = input.reshape((n_batches, sequence, dim));
        let x = x + self.attention.forward(x);
        let x = x.layer_norm(2, 1e-5);
        let x = x + self.ff.forward(x);
        x.layer_norm(2, 1e-5).reshape(input.shape())
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

    use super::TransformerEncoderBlock;
    #[test]
    fn test_transformer_encoder_block() {
        let mut cx = Graph::new();
        let model = TransformerEncoderBlock::new(3, 4, 1, &mut cx);
        model
            .attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .attention
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

        let a = cx
            .tensor(('s', 3))
            .set_dyn(vec![-1., 2., 3., 3., 3., -1.], (2, 3));
        let b = model.forward(a).retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
            d_dev
                .build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>(
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
        d_model.norm1.epsilon = 1e-5;
        d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.epsilon = 1e-5;
        let d_a = d_dev.tensor_from_vec(
            vec![-1., 2., 3., 3., 3., -1.],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward(d_a);

        assert_close(&b.data(), &d_b.as_vec());
    }
}
