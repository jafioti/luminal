use luminal::prelude::*;

mod attention;
pub use attention::*;
mod decoder;
pub use decoder::*;
mod encoder;
pub use encoder::*;

pub struct Transformer<
    const DIM: usize,
    const FF: usize,
    const ENC_HEADS: usize,
    const DEC_HEADS: usize,
    const ENC_LAYERS: usize,
    const DEC_LAYERS: usize,
> {
    pub encoder: encoder::TransformerEncoder<DIM, FF, ENC_HEADS, ENC_LAYERS>,
    pub decoder: decoder::TransformerDecoder<DIM, FF, DEC_HEADS, DEC_LAYERS>,
}

impl<
        const DIM: usize,
        const FF: usize,
        const ENC_HEADS: usize,
        const DEC_HEADS: usize,
        const ENC_LAYERS: usize,
        const DEC_LAYERS: usize,
    > InitModule for Transformer<DIM, FF, ENC_HEADS, DEC_HEADS, ENC_LAYERS, DEC_LAYERS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            encoder: InitModule::initialize(cx),
            decoder: InitModule::initialize(cx),
        }
    }
}

impl<
        const DIM: usize,
        const FF: usize,
        const ENC_HEADS: usize,
        const DEC_HEADS: usize,
        const ENC_LAYERS: usize,
        const DEC_LAYERS: usize,
    > SerializeModule for Transformer<DIM, FF, ENC_HEADS, DEC_HEADS, ENC_LAYERS, DEC_LAYERS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("encoder", &self.encoder);
        s.module("decoder", &self.decoder);
    }
}

// Single Sequence
impl<
        const DIM: usize,
        const FF: usize,
        const ENC_HEADS: usize,
        const DEC_HEADS: usize,
        const ENC_LAYERS: usize,
        const DEC_LAYERS: usize,
        S1: Dimension,
        S2: Dimension,
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
    use rand::{thread_rng, Rng};

    use super::Transformer;
    #[test]
    fn test_transformer_full() {
        let mut cx = Graph::new();
        let model: Transformer<3, 4, 1, 1, 1, 1> = InitModule::initialize(&mut cx);
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
        model.encoder.modules[0]
            .attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.encoder.modules[0]
            .attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model.encoder.modules[0]
            .attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model.encoder.modules[0]
            .attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model.encoder.modules[0]
            .ff
            .0
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model.encoder.modules[0]
            .ff
            .2
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

        let a = cx.tensor::<(Dyn<'d'>, luminal::shape::Const<3>)>();
        let e = cx.tensor::<(Dyn<'e'>, luminal::shape::Const<3>)>();
        let b = model.forward((a, e));

        a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], &[2, 3]);
        e.set_dyn(vec![-1., 2., 3., 3., 3., -1., -1., 2., 3.], &[3, 3]);
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

    #[test]
    fn test_serialization() {
        let mut rng = thread_rng();
        let enc_data = (0..(24 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();
        let trg_data = (0..(20 * 32)).map(|_| rng.gen()).collect::<Vec<f32>>();

        let mut cx = Graph::new();
        let model: Transformer<32, 5, 4, 4, 3, 2> = InitModule::initialize(&mut cx);
        let enc = cx.tensor::<R2<24, 32>>().set(enc_data.clone()).keep();
        let trg = cx.tensor::<R2<20, 32>>().set(trg_data.clone()).keep();
        let mut out1 = model.forward((trg, enc)).retrieve();
        cx.compile(GenericCompiler::default(), &mut out1);

        cx.execute_no_delete();

        let param_dict = ParamDictSaver.save(&model, &mut cx);
        let out1 = out1.data();

        let mut cx = Graph::new();
        let model: Transformer<32, 5, 4, 4, 3, 2> = InitModule::initialize(&mut cx);
        ParamDictLoader::new(param_dict).load(&model, &mut cx);
        let enc = cx.tensor::<R2<24, 32>>().set(enc_data);
        let trg = cx.tensor::<R2<20, 32>>().set(trg_data);
        let mut out2 = model.forward((trg, enc)).retrieve();

        cx.compile(GenericCompiler::default(), &mut out2);
        cx.execute();

        assert_close(&out1, &out2.data());
    }
}
