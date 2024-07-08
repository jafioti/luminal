use luminal::prelude::*;

mod attention;
pub use attention::*;
mod decoder;
pub use decoder::*;
mod encoder;
pub use encoder::*;

pub struct Transformer {
    pub encoder: encoder::TransformerEncoder,
    pub decoder: decoder::TransformerDecoder,
}

impl Transformer {
    pub fn new(
        dim: usize,
        ff: usize,
        enc_heads: usize,
        dec_heads: usize,
        enc_layers: usize,
        dec_layers: usize,
        cx: &mut Graph,
    ) -> Self {
        Self {
            encoder: TransformerEncoder::new(dim, ff, enc_heads, enc_layers, cx),
            decoder: TransformerDecoder::new(dim, ff, dec_heads, dec_layers, cx),
        }
    }
}

impl SerializeModule for Transformer {
    fn serialize(&self, s: &mut Serializer) {
        s.module("encoder", &self.encoder);
        s.module("decoder", &self.decoder);
    }
}

// Single Sequence
impl Module<(GraphTensor, GraphTensor)> for Transformer {
    type Output = GraphTensor;

    fn forward(&self, (input, target): (GraphTensor, GraphTensor)) -> Self::Output {
        let encoded = self.encoder.forward(input);
        self.decoder.forward((target, encoded))
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

    use super::Transformer;
    #[test]
    fn test_transformer_full() {
        let mut cx = Graph::new();
        let model = Transformer::new(3, 4, 1, 1, 1, 1, &mut cx);
        model.decoder.layers[0]
            .self_attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.decoder.layers[0]
            .self_attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model.decoder.layers[0]
            .self_attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model.decoder.layers[0]
            .self_attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.decoder.layers[0]
            .cross_attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.decoder.layers[0]
            .cross_attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model.decoder.layers[0]
            .cross_attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model.decoder.layers[0]
            .cross_attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.decoder.layers[0]
            .ff
            .0
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model.decoder.layers[0]
            .ff
            .2
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);
        model.encoder.layers[0]
            .attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.encoder.layers[0]
            .attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model.encoder.layers[0]
            .attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model.encoder.layers[0]
            .attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.encoder.layers[0]
            .ff
            .0
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model.encoder.layers[0]
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
        let mut d_model: dfdx::nn::modules::Transformer<3, 1, 1, 1, 4, f32, Cpu> =
            d_dev.build_module::<dfdx::nn::modules::builders::Transformer<3, 1, 1, 1, 4>, f32>();
        d_model.decoder.0.modules[0]
            .self_attn
            .w_k
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .self_attn
            .w_v
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .self_attn
            .w_q
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .self_attn
            .w_o
            .bias
            .copy_from(&[0., 0., 0.]);
        d_model.decoder.0.modules[0].self_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].self_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].self_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].self_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0]
            .mh_attn
            .w_k
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .mh_attn
            .w_v
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .mh_attn
            .w_q
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.decoder.0.modules[0]
            .mh_attn
            .w_o
            .bias
            .copy_from(&[0., 0., 0.]);
        d_model.decoder.0.modules[0].mh_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].mh_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].mh_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].mh_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].ff.0 .0.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        d_model.decoder.0.modules[0].ff.0 .0.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
        d_model.decoder.0.modules[0].ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.decoder.0.modules[0].ff.0 .2.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm1.gamma =
            d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm2.gamma =
            d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm3.gamma =
            d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm1.beta =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm2.beta =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm3.beta =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.decoder.0.modules[0].norm1.epsilon = 1e-5;
        d_model.decoder.0.modules[0].norm2.epsilon = 1e-5;
        d_model.decoder.0.modules[0].norm3.epsilon = 1e-5;
        d_model.encoder.modules[0]
            .self_attn
            .w_k
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.encoder.modules[0]
            .self_attn
            .w_v
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.encoder.modules[0]
            .self_attn
            .w_q
            .bias
            .copy_from(&[0.0, 0.0, 0.0]);
        d_model.encoder.modules[0]
            .self_attn
            .w_o
            .bias
            .copy_from(&[0., 0., 0.]);
        d_model.encoder.modules[0].self_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.encoder.modules[0].self_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.encoder.modules[0].self_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.encoder.modules[0].self_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.encoder.modules[0].ff.0 .0.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        d_model.encoder.modules[0].ff.0 .0.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
        d_model.encoder.modules[0].ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.encoder.modules[0].ff.0 .2.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.encoder.modules[0].norm1.gamma =
            d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.encoder.modules[0].norm2.gamma =
            d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.encoder.modules[0].norm1.epsilon = 1e-5;
        d_model.encoder.modules[0].norm2.beta =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.encoder.modules[0].norm1.beta =
            d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.encoder.modules[0].norm2.epsilon = 1e-5;
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
