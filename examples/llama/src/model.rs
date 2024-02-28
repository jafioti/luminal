#![allow(clippy::type_complexity)]
use std::{marker::PhantomData, ops::Mul};

// LLaMa 1 7B Config
pub const VOCAB: usize = 32_000;
pub const HEAD_DIM: usize = 128;
pub const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 11008;
pub const HEADS: usize = 32;
pub const LAYERS: usize = 32;

use luminal::{
    nn::{embedding::Embedding, norm::RMSNorm},
    prelude::*,
    shape::symbolic::{BigExpression, Expression},
};

// Full LLaMa model implementation, heavily based off of https://github.com/coreylowman/llama-dfdx/blob/main/src/modeling.rs

pub type KVCache<Batch, Seq> = (
    GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
);

pub struct Mlp<const I: usize, const H: usize> {
    pub gate_proj: GraphTensor<(Const<I>, Const<H>)>,
    pub down_proj: GraphTensor<(Const<H>, Const<I>)>,
    pub up_proj: GraphTensor<(Const<I>, Const<H>)>,
}

impl<Sh: Shape, Im: Shape, const I: usize, const H: usize> Module<GraphTensor<Sh>> for Mlp<I, H>
where
    GraphTensor<Sh>: Matmul<R2<H, I>, Output = GraphTensor<Im>>,
    GraphTensor<Im>: Matmul<R2<I, H>, Output = GraphTensor<Sh>>,
{
    type Output = GraphTensor<Sh>;

    fn forward(&self, input: GraphTensor<Sh>) -> Self::Output {
        let gate = input.matmul(self.gate_proj.permute()).swish();
        let up = input.matmul(self.up_proj.permute()) * gate;
        up.matmul(self.down_proj.permute())
    }
}

impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            gate_proj: cx.named_tensor("Gate Weight"),
            up_proj: cx.named_tensor("Up Weight"),
            down_proj: cx.named_tensor("Down Weight"),
        }
    }
}

impl<const I: usize, const H: usize> SerializeModule for Mlp<I, H> {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("gate_proj/weight", self.gate_proj);
        s.tensor("up_proj/weight", self.up_proj);
        s.tensor("down_proj/weight", self.down_proj);
    }
}

pub struct RotaryEmbedding {
    pub inv_freq: GraphTensor<R1<HEAD_DIM_OVER_2>>,
}

impl<Batch: Dimension, Seq: Dimension>
    Module<(
        GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
        BigExpression,
    )> for RotaryEmbedding
{
    type Output = GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>;

    fn forward(
        &self,
        (inp, prev_seq): (
            GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
            BigExpression,
        ),
    ) -> Self::Output {
        let (sin, cos) = self.get_sincos::<Seq>(prev_seq);
        (Self::rotate_half(inp) * sin.expand()) + (inp * cos.expand())
    }
}

impl RotaryEmbedding {
    fn get_sincos<Seq: Dimension>(
        &self,
        prev_seq: BigExpression,
    ) -> (
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
    ) {
        let t = self.inv_freq.graph().arange::<Seq>() + prev_seq;
        let freqs = t.expand::<(Seq, Const<1>), _>().matmul(
            self.inv_freq
                .expand::<(Const<1>, Const<HEAD_DIM_OVER_2>), _>(),
        );
        let emb = freqs.concat_along::<(Seq, Const<HEAD_DIM>), Axis<1>, _>(freqs);
        (emb.sin().reshape(), emb.cos().reshape())
    }

    fn rotate_half<Batch: Dimension, Seq: Dimension>(
        x: GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
    ) -> GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)> {
        let x1 = x
            .slice((.., .., .., ..Expression::from(HEAD_DIM_OVER_2)))
            .contiguous();
        let x2 = x
            .slice((.., .., .., Expression::from(HEAD_DIM_OVER_2)..))
            .contiguous();
        (-x2).concat_along::<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>), Axis<3>, _>(x1)
    }
}

impl InitModule for RotaryEmbedding {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            inv_freq: cx.named_tensor("Inv Freq"),
        }
    }
}

impl SerializeModule for RotaryEmbedding {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("inv_freq", self.inv_freq);
    }
}

pub struct Attention {
    pub q_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub k_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub v_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub o_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub rotary_embed: RotaryEmbedding,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Option<KVCache<Batch, PrevSeq>>,
        PhantomData<TotSeq>,
    )> for Attention
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq>,
    );

    fn forward(
        &self,
        (x, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            Option<KVCache<Batch, PrevSeq>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        let queries = x
            .matmul(self.q_proj.permute())
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let keys = x
            .matmul(self.k_proj.permute())
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let values = x
            .matmul(self.v_proj.permute())
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let queries = self
            .rotary_embed
            .forward((queries.permute(), PrevSeq::const_size().into()));
        let keys = self
            .rotary_embed
            .forward((keys, PrevSeq::const_size().into()));

        let (keys, values) = if let Some((k_cache, v_cache)) = cache {
            (
                k_cache.concat_along::<_, Axis<2>, _>(keys),
                v_cache.concat_along::<_, Axis<2>, _>(values),
            )
        } else {
            (keys.realize(), values.contiguous().realize())
        };

        let mut weights = queries
            .matmul(keys.permute())
            .mul((HEAD_DIM as f64).sqrt().recip() as f32);
        let attention_mask = self.k_proj.graph().triu::<CurSeq>(1) * f16::MIN.to_f32();
        weights += attention_mask
            .pad::<(CurSeq, TotSeq), _, _>(&[
                (0.into(), Expression::from(0)),
                (TotSeq::const_size() - CurSeq::const_size(), 0.into()),
            ])
            .expand();

        let outputs = weights
            .softmax::<3>()
            .matmul(values)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, CurSeq, Const<HIDDEN>)>();
        (
            outputs.matmul(self.o_proj.permute()),
            (keys.contiguous(), values.contiguous()),
        )
    }
}

impl InitModule for Attention {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Query Weight"),
            k_proj: cx.named_tensor("Key Weight"),
            v_proj: cx.named_tensor("Value Weight"),
            o_proj: cx.named_tensor("Output Weight"),
            rotary_embed: InitModule::initialize(cx),
        }
    }
}

impl SerializeModule for Attention {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("q_proj/weight", self.q_proj);
        s.tensor("k_proj/weight", self.k_proj);
        s.tensor("v_proj/weight", self.v_proj);
        s.tensor("o_proj/weight", self.o_proj);
        s.module("rotary_emb", &self.rotary_embed);
    }
}

pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: Mlp<INTERMEDIATE, HIDDEN>,
    pub input_layer_norm: RMSNorm<HIDDEN>,
    pub post_attention_layer_norm: RMSNorm<HIDDEN>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Option<KVCache<Batch, PrevSeq>>,
        PhantomData<TotSeq>,
    )> for TransformerBlock
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (mut x, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            Option<KVCache<Batch, PrevSeq>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Attention
        let normed = self.input_layer_norm.forward(x);
        let (y, cache) = self
            .self_attn
            .forward((normed, cache, PhantomData::<TotSeq>));

        // Residual Addition
        x += y;

        // Feed Forward
        let y = self.mlp.forward(self.post_attention_layer_norm.forward(x));

        // Residual Addition
        (x + y, cache)
    }
}

impl InitModule for TransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            self_attn: InitModule::initialize(cx),
            mlp: InitModule::initialize(cx),
            input_layer_norm: InitModule::initialize(cx),
            post_attention_layer_norm: InitModule::initialize(cx),
        }
    }
}

impl SerializeModule for TransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.self_attn);
        s.module("mlp", &self.mlp);
        s.module("input_layernorm", &self.input_layer_norm);
        s.module("post_attention_layernorm", &self.post_attention_layer_norm);
    }
}

pub struct Llama {
    // Token embeddings
    pub embedding: Embedding<VOCAB, HIDDEN>,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Final Norm layer
    pub norm: RMSNorm<HIDDEN>,
    // LM Head Layer
    pub lm_head: GraphTensor<R2<VOCAB, HIDDEN>>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq)>,
        Option<Vec<KVCache<Batch, PrevSeq>>>,
        PhantomData<TotSeq>,
    )> for Llama
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<VOCAB>)>,
        Vec<KVCache<Batch, TotSeq>>,
    );
    fn forward(
        &self,
        (input, cache, _): (
            GraphTensor<(Batch, CurSeq)>,
            Option<Vec<KVCache<Batch, PrevSeq>>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Embed tokens
        let mut x = self.embedding.forward(input);

        // Run through layers and collect new caches
        let mut new_caches = vec![];
        let mut new_cache;
        for (i, layer) in self.layers.iter().enumerate() {
            (x, new_cache) =
                layer.forward((x, cache.as_ref().map(|c| c[i]), PhantomData::<TotSeq>));
            new_caches.push(new_cache);
        }
        // Run through last norm and output projection
        let output = self.norm.forward(x);
        let output = output.matmul(self.lm_head.permute());

        (output, new_caches)
    }
}

impl InitModule for Llama {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            norm: InitModule::initialize(cx),
            embedding: InitModule::initialize(cx),
            layers: (0..LAYERS).map(|_| InitModule::initialize(cx)).collect(),
            lm_head: cx.named_tensor("LM Head"),
        }
    }
}

impl SerializeModule for Llama {
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/norm", &self.norm);
        s.module("model/embed_tokens", &self.embedding);
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("model/layers/{i}"), l);
        }
        s.tensor("lm_head/weight", self.lm_head);
    }
}
