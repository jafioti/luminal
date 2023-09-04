#![allow(clippy::type_complexity)]
use std::ops::{Add, Mul};

use luminal::{
    nn::{activation::RMSNorm, embedding::Embedding},
    op::Function,
    prelude::{movement::TryConcatAlong, *},
};
use rand::{thread_rng, Rng};

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

impl<const I: usize, const H: usize, B: Dimension, S: Dimension>
    Module<GraphTensor<(B, S, Const<H>)>> for Mlp<I, H>
{
    type Output = GraphTensor<(B, S, Const<H>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<H>)>) -> Self::Output {
        let gate = input.matmul(self.gate_proj.permute());
        let gate = gate.sigmoid() * gate;
        let up = input.matmul(self.up_proj.permute()) * gate;
        up.matmul(self.down_proj.permute())
    }
}

impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            gate_proj: cx.new_tensor("Gate Weight"),
            up_proj: cx.new_tensor("Up Weight"),
            down_proj: cx.new_tensor("Down Weight"),
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
        PrevSeq: Dimension,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    >
    Module<(
        GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
        GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
        Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
    )> for RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = (
        GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
        GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
    );

    fn forward(
        &self,
        (q, k, cache): (
            GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
            GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
            Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
        ),
    ) -> Self::Output {
        let (sin, cos) = self.get_sincos::<Batch, NUM_HEADS, Seq, PrevSeq>(q, cache);
        let sin = sin.expand();
        let cos = cos.expand();
        let q_embed = (Self::rotate_half(q) * sin) + (q * cos);
        let k_embed = (Self::rotate_half(k) * sin) + (k * cos);
        (q_embed, k_embed)
    }
}

impl<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize>
    RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn get_sincos<Batch: Dimension, const NUM_HEADS: usize, Seq: Dimension, PrevSeq: Dimension>(
        &self,
        seq_tensor: GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
        cache: Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
    ) -> (
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
        GraphTensor<(Seq, Const<HEAD_DIM>)>,
    ) {
        let graph = unsafe { self.inv_freq.graph_ref.as_mut().unwrap() };
        let has_cache = cache.is_some();
        let mut op = graph
            .add_op(Function(
                "ARange".to_string(),
                Box::new(move |inp| {
                    let offset = if has_cache {
                        inp[1].1.shape()[2].to_usize().unwrap()
                    } else {
                        0
                    };
                    Tensor {
                        data: Box::new(
                            (0..inp[0].1.shape()[2].to_usize().unwrap())
                                .map(|i| (i + offset) as f32)
                                .collect::<Vec<_>>(),
                        ),
                    }
                }),
            ))
            .input(seq_tensor.id, seq_tensor.shape);
        if has_cache {
            op = op.input(cache.unwrap().0.id, cache.unwrap().0.shape);
        }
        let t: GraphTensor<(Seq,)> =
            GraphTensor::from_id(op.finish(), <(Seq,)>::to_tracker(), graph);
        let freqs = t
            .expand::<(Seq, Const<1>), _>()
            .matmul(
                self.inv_freq
                    .expand::<(Const<1>, Const<HEAD_DIM_OVER_2>), _>(),
            )
            .realize::<(Seq, Dyn<'-'>)>();
        let emb = (freqs, freqs).concat_along(Axis::<1>);
        (emb.sin().realize(), emb.cos().realize())
    }

    fn rotate_half<Batch: Dimension, NumHeads: Dimension, Seq: Dimension>(
        x: GraphTensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>)>,
    ) -> GraphTensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>)> {
        let x1 = x.slice((.., .., .., ..HEAD_DIM_OVER_2));
        let x2 = x.slice((.., .., .., HEAD_DIM_OVER_2..));
        (-x2, x1).concat_along(Axis::<3>).realize()
    }
}

impl<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize> InitModule
    for RotaryEmbedding<HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            inv_freq: cx.new_tensor("Inv Freq"),
        };
        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        s.inv_freq.set(
            (0..HEAD_DIM_OVER_2)
                .map(|_| rng.gen_range(-1_f32..1_f32))
                .collect::<Vec<_>>(),
        );
        s
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

fn attn_forward<
    const NUM_HEADS: usize,
    const HIDDEN: usize,
    const HEAD_DIM: usize,
    const HEAD_DIM_OVER_2: usize,
    Batch: Dimension,
    Seq: Dimension,
    PrevSeq: Dimension,
>(
    attn: &Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>,
    x: GraphTensor<(Batch, Seq, Const<HIDDEN>)>,
    cache: Option<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
) -> (
    GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>,
) {
    let q = x
        .matmul(attn.q_proj.permute())
        .dyn_reshape::<(Batch, Seq, Const<NUM_HEADS>, Const<HEAD_DIM>)>(vec![
            Batch::const_size(),
            Seq::const_size(),
            Dim::Known(NUM_HEADS),
            Dim::Known(HEAD_DIM),
        ])
        .permute::<_, Axes4<0, 2, 1, 3>>();

    let k = x
        .matmul(attn.k_proj.permute())
        .dyn_reshape::<(Batch, Seq, Const<NUM_HEADS>, Const<HEAD_DIM>)>(vec![
            Batch::const_size(),
            Seq::const_size(),
            Dim::Known(NUM_HEADS),
            Dim::Known(HEAD_DIM),
        ])
        .permute::<_, Axes4<0, 2, 1, 3>>();
    let v = x
        .matmul(attn.v_proj.permute())
        .dyn_reshape::<(Batch, Seq, Const<NUM_HEADS>, Const<HEAD_DIM>)>(vec![
            Batch::const_size(),
            Seq::const_size(),
            Dim::Known(NUM_HEADS),
            Dim::Known(HEAD_DIM),
        ])
        .permute::<_, Axes4<0, 2, 1, 3>>();
    let (q, k) = attn.rotary_embed.forward((
        q.realize::<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>)>(),
        k.realize(),
        cache,
    ));

    (q, k, v)
}

impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        Batch: Dimension,
        CurSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        GraphTensor<(CurSeq, CurSeq)>,
    )> for Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, CurSeq, NUM_HEADS, HEAD_DIM>,
    );

    fn forward(
        &self,
        (x, attn_mask): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            GraphTensor<(CurSeq, CurSeq)>,
        ),
    ) -> Self::Output {
        let (q, k, v) = attn_forward(
            self,
            x,
            Option::<KVCache<_, Dyn<'s'>, NUM_HEADS, HEAD_DIM>>::None,
        );
        let inv_head_scale = (HEAD_DIM as f64).sqrt().recip() as f32;
        let w = q
            .batch_matmul(k.permute())
            .mul(inv_head_scale)
            .add(attn_mask.expand())
            .softmax::<3>();

        let o = w
            .batch_matmul(v)
            .permute::<(Batch, CurSeq, Const<NUM_HEADS>, Const<HEAD_DIM>), _>()
            .dyn_reshape::<(Batch, CurSeq, Const<HIDDEN>)>(vec![
                Batch::const_size(),
                CurSeq::const_size(),
                Dim::Known(HIDDEN),
            ]);

        (o.matmul(self.o_proj.permute()), (k, v))
    }
}

// KV cache forward
impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > Attention<NUM_HEADS, HIDDEN, HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn forward_kv<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>(
        &self,
        (x, cache): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>,
        ),
    ) -> (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>,
    ) {
        let (q, k, v) = attn_forward(self, x, Some(cache));

        // Add KV cache
        let k = (
            cache
                .0
                .realize::<(Batch, Const<NUM_HEADS>, Dyn<'s'>, Const<HEAD_DIM>)>(),
            k.realize::<(Batch, Const<NUM_HEADS>, Dyn<'s'>, Const<HEAD_DIM>)>(),
        )
            .concat_along(Axis::<2>)
            .realize::<(Batch, Const<NUM_HEADS>, TotSeq, Const<HEAD_DIM>)>();
        let v = (
            cache
                .1
                .realize::<(Batch, Const<NUM_HEADS>, Dyn<'s'>, Const<HEAD_DIM>)>(),
            v.realize::<(Batch, Const<NUM_HEADS>, Dyn<'s'>, Const<HEAD_DIM>)>(),
        )
            .concat_along(Axis::<2>)
            .realize::<(Batch, Const<NUM_HEADS>, TotSeq, Const<HEAD_DIM>)>();

        let w = q
            .batch_matmul(k.permute())
            .mul((HEAD_DIM as f64).sqrt().recip() as f32) // Inv head scale
            .softmax::<3>();

        let o = w
            .batch_matmul(v)
            .permute::<(Batch, CurSeq, Const<NUM_HEADS>, Const<HEAD_DIM>), _>()
            .dyn_reshape::<(Batch, CurSeq, Const<HIDDEN>)>(vec![
                Batch::const_size(),
                CurSeq::const_size(),
                Dim::Known(HIDDEN),
            ]);

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
            q_proj: cx.new_tensor("Query Weight"),
            k_proj: cx.new_tensor("Key Weight"),
            v_proj: cx.new_tensor("Value Weight"),
            o_proj: cx.new_tensor("Output Weight"),
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
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        GraphTensor<(CurSeq, CurSeq)>,
    )> for DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, CurSeq, NUM_HEADS, HEAD_DIM>,
    );
    fn forward(
        &self,
        (x, attn_mask): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            GraphTensor<(CurSeq, CurSeq)>,
        ),
    ) -> Self::Output {
        let (y, kv_cache) = self
            .self_attn
            .forward((self.input_layer_norm.forward(x), attn_mask));
        let x = x + y;
        let y = self.mlp.forward(self.post_attention_layer_norm.forward(x));
        (x + y, kv_cache)
    }
}

// KV cache forward
impl<
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
    > DecoderLayer<NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2>
{
    fn forward_kv<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>(
        &self,
        (x, cache): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>,
        ),
    ) -> (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>,
    ) {
        let (y, kv_cache) = self
            .self_attn
            .forward_kv((self.input_layer_norm.forward(x), cache));
        let x = x + y;
        let y = self.mlp.forward(self.post_attention_layer_norm.forward(x));
        (x + y, kv_cache)
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

pub struct Llama<
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
    pub graph_ref: *mut Graph,
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
    > Module<GraphTensor<(Batch, CurSeq)>>
    for Llama<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Vec<KVCache<Batch, CurSeq, NUM_HEADS, HEAD_DIM>>,
    );
    fn forward(&self, input: GraphTensor<(Batch, CurSeq)>) -> Self::Output {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let attn_mask: GraphTensor<(CurSeq, CurSeq)> = GraphTensor::from_id(
            graph
                .add_op(Function(
                    "AttentionMask".to_string(),
                    Box::new(|inp| {
                        let seq_len = inp[0].1.shape()[1].to_usize().unwrap();
                        let mut data = vec![0.; seq_len * seq_len];
                        for i in 0..seq_len {
                            for j in (i + 1)..seq_len {
                                data[i * seq_len + j] = f32::NEG_INFINITY;
                            }
                        }
                        Tensor {
                            data: Box::new(data),
                        }
                    }),
                ))
                .input(input.id, input.shape)
                .finish(),
            ShapeTracker::new(&[CurSeq::const_size(), CurSeq::const_size()]),
            graph,
        );

        let mut hidden_states = self.embed_tokens.forward(input);
        let mut caches = vec![];
        for layer_i in &self.layers {
            let (new_hidden_states, kv_cache) = layer_i.forward((hidden_states, attn_mask));
            hidden_states = new_hidden_states;
            caches.push(kv_cache);
        }
        (self.norm.forward(hidden_states), caches)
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
    > Llama<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    pub fn forward_kv<
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >(
        &self,
        (input, caches): (
            GraphTensor<(Batch, CurSeq)>,
            Vec<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
        ),
    ) -> (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Vec<KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>>,
    ) {
        let mut hidden_states = self.embed_tokens.forward(input);
        let mut new_caches = vec![];
        for (layer_i, cache) in self.layers.iter().zip(caches.into_iter()) {
            let (new_hidden_states, kv_cache) = layer_i.forward_kv((hidden_states, cache));
            hidden_states = new_hidden_states;
            new_caches.push(kv_cache);
        }
        (self.norm.forward(hidden_states), new_caches)
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
    for Llama<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            norm: InitModule::initialize(cx),
            embed_tokens: InitModule::initialize(cx),
            layers: (0..LAYERS).map(|_| InitModule::initialize(cx)).collect(),
            graph_ref: cx,
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
    for Llama<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("norm", &self.norm);
        s.module("embed_tokens", &self.embed_tokens);
        for (i, l) in self.layers.iter().enumerate() {
            s.module(&format!("layers/{i}"), l);
        }
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
    pub llama: Llama<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>,
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
    > Module<GraphTensor<(Batch, CurSeq)>>
    for LlamaForCausalLM<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<VOCAB>)>,
        Vec<KVCache<Batch, CurSeq, NUM_HEADS, HEAD_DIM>>,
    );
    fn forward(&self, input: GraphTensor<(Batch, CurSeq)>) -> Self::Output {
        let (hidden_states, caches) = self.llama.forward(input);
        (hidden_states.matmul(self.lm_head.permute()), caches)
    }
}

// KV cache forward
impl<
        const VOCAB: usize,
        const NUM_HEADS: usize,
        const HIDDEN: usize,
        const INTERMEDIATE: usize,
        const HEAD_DIM: usize,
        const HEAD_DIM_OVER_2: usize,
        const LAYERS: usize,
    > LlamaForCausalLM<VOCAB, NUM_HEADS, HIDDEN, INTERMEDIATE, HEAD_DIM, HEAD_DIM_OVER_2, LAYERS>
{
    pub fn forward_kv<
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >(
        &self,
        (input, caches): (
            GraphTensor<(Batch, CurSeq)>,
            Vec<KVCache<Batch, PrevSeq, NUM_HEADS, HEAD_DIM>>,
        ),
    ) -> (
        GraphTensor<(Batch, CurSeq, Const<VOCAB>)>,
        Vec<KVCache<Batch, TotSeq, NUM_HEADS, HEAD_DIM>>,
    ) {
        let (hidden_states, caches) = self.llama.forward_kv((input, caches));
        (hidden_states.matmul(self.lm_head.permute()), caches)
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
            llama: InitModule::initialize(cx),
            lm_head: cx.new_tensor("LM Head"),
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
        s.module("model", &self.llama);
        s.tensor("lm_head/weight", self.lm_head);
    }
}
