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

use half::f16;
use luminal::{
    nn::{embedding::Embedding, norm::RMSNorm},
    prelude::*,
    shape::symbolic::{BigExpression, Expression},
};

// Full LLaMa model implementation, heavily based off of https://github.com/coreylowman/llama-dfdx/blob/main/src/modeling.rs

pub type KVCache<Batch, Seq, const NUM_HEADS: usize, const HEAD_DIM: usize> = (
    GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
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

pub struct RotaryEmbedding<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize> {
    pub inv_freq: GraphTensor<R1<HEAD_DIM_OVER_2>>,
}

impl<
        Batch: Dimension,
        const NUM_HEADS: usize,
        Seq: Dimension,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    >
    Module<(
        GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
        BigExpression,
    )> for RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>;

    fn forward(
        &self,
        (inp, prev_seq): (
            GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
            BigExpression,
        ),
    ) -> Self::Output {
        let (sin, cos) = self.get_sincos::<NUM_HEADS, Seq>(prev_seq);
        (Self::rotate_half(inp) * sin.expand()) + (inp * cos.expand())
    }
}

impl<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize>
    RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn get_sincos<const NUM_HEADS: usize, Seq: Dimension>(
        &self,
        prev_seq: BigExpression,
    ) -> (
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
    ) {
        let t = self.inv_freq.graph().arange::<Seq>()
            + self.inv_freq.graph().constant_expr(prev_seq).expand();
        let freqs = t.expand::<(Seq, Const<1>), _>().matmul(
            self.inv_freq
                .expand::<(Const<1>, Const<HEAD_DIM_OVER_2>), _>(),
        );
        let emb = freqs.concat_along::<(Seq, Const<HEAD_DIM>), Axis<1>, _>(freqs);
        (emb.sin().reshape(), emb.cos().reshape())
    }

    fn rotate_half<Batch: Dimension, NumHeads: Dimension, Seq: Dimension>(
        x: GraphTensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>)>,
    ) -> GraphTensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>)> {
        let x1 = x
            .slice((.., .., .., ..Expression::from(HEAD_DIM_OVER_2)))
            .contiguous();
        let x2 = x
            .slice((.., .., .., Expression::from(HEAD_DIM_OVER_2)..))
            .contiguous();
        (-x2).concat_along::<(Batch, NumHeads, Seq, Const<HEAD_DIM>), Axis<3>, _>(x1)
    }
}

impl<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize> InitModule
    for RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            inv_freq: cx.named_tensor("Inv Freq"),
        }
    }
}

impl<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize> SerializeModule
    for RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("inv_freq", self.inv_freq);
    }
}

pub struct Attention<
    const NUM_HEADS: usize,
    const HIDDEN: usize,
    const HEAD_DIM: usize,
    const HEAD_DIM_OVER_2: usize,
> {
    pub q_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub k_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub v_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub o_proj: GraphTensor<(Const<HIDDEN>, Const<HIDDEN>)>,
    pub rotary_embed: RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>,
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
        PhantomData<TotSeq>,
    )> for Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>,
    );

    fn forward(
        &self,
        (x, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        let q = x
            .matmul(self.q_proj.permute())
            .reshape::<(Batch, CurSeq, Const<NUM_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let k = x
            .matmul(self.k_proj.permute())
            .reshape::<(Batch, CurSeq, Const<NUM_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let v = x
            .matmul(self.v_proj.permute())
            .reshape::<(Batch, CurSeq, Const<NUM_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let q = self
            .rotary_embed
            .forward((q.permute(), PrevSeq::const_size().into()));
        let k = self.rotary_embed.forward((k, PrevSeq::const_size().into()));

        let (k, v) = if let Some(cache) = cache {
            // Add KV cache
            let k = cache
                .0
                .concat_along::<(Batch, Const<NUM_HEADS>, TotSeq, Const<HEAD_DIM>), Axis<2>, _>(k);
            let v = cache
                .1
                .concat_along::<(Batch, Const<NUM_HEADS>, TotSeq, Const<HEAD_DIM>), Axis<2>, _>(v);
            (k, v)
        } else {
            (k.realize(), v.realize())
        };

        let mut w = q
            .matmul(k.permute())
            .mul((HEAD_DIM as f64).sqrt().recip() as f32);
        // We only mask on a non-kv cache pass
        if cache.is_none() {
            let attention_mask = self.k_proj.graph().triu::<CurSeq>(1) * f16::MIN.to_f32();
            w += attention_mask.realize::<(CurSeq, TotSeq)>().expand(); // CurSeq and TotSeq are guarenteed to be the same size here
        }
        w = w.softmax::<3>();

        let o = w
            .matmul(v)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, CurSeq, Const<HIDDEN>)>();
        (o.matmul(self.o_proj.permute()), (k, v))
    }
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > InitModule for Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>
{
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

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > SerializeModule for Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("q_proj/weight", self.q_proj);
        s.tensor("k_proj/weight", self.k_proj);
        s.tensor("v_proj/weight", self.v_proj);
        s.tensor("o_proj/weight", self.o_proj);
        s.module("rotary_emb", &self.rotary_embed);
    }
}

pub struct DecoderLayer<
    const NUM_HEADS: usize,
    const HIDDEN: usize,
    const INTERMEDIATE: usize,
    const HEAD_DIM: usize,
    const HEAD_DIM_OVER_2: usize,
> {
    pub self_attn: Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>,
    pub mlp: Mlp<INTERMEDIATE, HIDDEN>,
    pub input_layer_norm: RMSNorm<HIDDEN>,
    pub post_attention_layer_norm: RMSNorm<HIDDEN>,
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
        PhantomData<TotSeq>,
    )> for DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>,
    );
    fn forward(
        &self,
        (x, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        let normed = self.input_layer_norm.forward(x);
        let (y, cache) = self
            .self_attn
            .forward((normed, cache, PhantomData::<TotSeq>));
        let x = x + y;
        let y = self.mlp.forward(self.post_attention_layer_norm.forward(x));
        (x + y, cache)
    }
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > InitModule for DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            self_attn: InitModule::initialize(cx),
            mlp: InitModule::initialize(cx),
            input_layer_norm: InitModule::initialize(cx),
            post_attention_layer_norm: InitModule::initialize(cx),
        }
    }
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > SerializeModule for DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.self_attn);
        s.module("mlp", &self.mlp);
        s.module("input_layernorm", &self.input_layer_norm);
        s.module("post_attention_layernorm", &self.post_attention_layer_norm);
    }
}

pub struct LlamaForCausalLM<
    const VOCAB: usize,
    const NUM_HEADS: usize,
    const HIDDEN: usize,
    const INTERMEDIATE: usize,
    const HEAD_DIM: usize,
    const HEAD_DIM_OVER_2: usize,
    const LAYERS: usize,
> {
    pub embed_tokens: Embedding<VOCAB, HIDDEN>,
    pub layers: Vec<DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>>,
    pub norm: RMSNorm<HIDDEN>,
    pub lm_head: GraphTensor<(Const<VOCAB>, Const<HIDDEN>)>,
}

impl<
        const VOCAB: usize,
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        const LAYERS: usize,
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq)>,
        Option<Vec<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>>,
        PhantomData<TotSeq>,
    )>
    for LlamaForCausalLM<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<VOCAB>)>,
        Vec<KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>>,
    );
    fn forward(
        &self,
        (input, caches, _): (
            GraphTensor<(Batch, CurSeq)>,
            Option<Vec<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        let mut hidden_states = self.embed_tokens.forward(input);
        let mut new_caches = Vec::with_capacity(LAYERS);
        for (i, layer_i) in self.layers.iter().enumerate() {
            let (new_hidden_states, (k_cache, v_cache)) = layer_i.forward((
                hidden_states,
                caches.as_ref().map(|v| v[i]),
                PhantomData::<TotSeq>,
            ));
            hidden_states = new_hidden_states;
            new_caches.push((k_cache.contiguous(), v_cache.contiguous()));
        }
        hidden_states = self.norm.forward(hidden_states);
        (hidden_states.matmul(self.lm_head.permute()), new_caches)
    }
}

impl<
        const VOCAB: usize,
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        const LAYERS: usize,
    > InitModule
    for LlamaForCausalLM<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            norm: InitModule::initialize(cx),
            embed_tokens: InitModule::initialize(cx),
            layers: (0..LAYERS).map(|_| InitModule::initialize(cx)).collect(),
            lm_head: cx.named_tensor("LM Head"),
        }
    }
}

impl<
        const VOCAB: usize,
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        const LAYERS: usize,
    > SerializeModule
    for LlamaForCausalLM<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/norm", &self.norm);
        s.module("model/embed_tokens", &self.embed_tokens);
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("model/layers/{i}"), l);
        }
        s.tensor("lm_head/weight", self.lm_head);
    }
}
