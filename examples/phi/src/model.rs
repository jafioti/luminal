use std::marker::PhantomData;

use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Embedding, LayerNorm, PermutedLinear};

// Llama3 8B Config
pub const VOCAB_SIZE: usize = 32064;
pub const HIDDEN_DIM: usize = 3072;
pub const NUM_LAYERS: usize = 32;
pub const N_HEADS: usize = 32;
pub const MLP_DIM: usize = 8192;

pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS;
pub const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
pub const ATTN_PROJ_DIM: usize = HEAD_DIM * N_HEADS;

pub type KVCache<Batch, Seq> = (
    GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
);

pub struct Mlp<const I: usize, const H: usize> {
    pub gate_proj: PermutedLinear<H, I>,
    pub down_proj: PermutedLinear<I, H>,
    pub up_proj: PermutedLinear<H, I>,
}

impl<const I: usize, const H: usize, Batch: Dimension, Batch1: Dimension>
    Module<GraphTensor<(Batch, Batch1, Const<H>)>> for Mlp<I, H>
{
    type Output = GraphTensor<(Batch, Batch1, Const<H>)>;

    fn forward(&self, input: GraphTensor<(Batch, Batch1, Const<H>)>) -> Self::Output {
        let gate = self.gate_proj.forward(input).swish();
        let up = self.up_proj.forward(input) * gate;
        self.down_proj.forward(up)
    }
}

impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            gate_proj: PermutedLinear::named("Gate", false, cx),
            up_proj: PermutedLinear::named("Up", false, cx),
            down_proj: PermutedLinear::named("Down", false, cx),
        }
    }
}

impl<const I: usize, const H: usize> SerializeModule for Mlp<I, H> {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ffn_gate", &self.gate_proj);
        s.module("ffn_up", &self.up_proj);
        s.module("ffn_down", &self.down_proj);
    }
}

fn apply_rotary_embeddings_ggml<const N_HEADS: usize, Batch: Dimension, Seq: Dimension>(
    input: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
    prev_seq: BigExpression,
) -> GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)> {
    // Get freqs
    let freqs = (input.graph().arange::<Const<HEAD_DIM_OVER_2>>() * 2.0) / (HEAD_DIM as f32);
    let freqs = 10_000_f32.pow(freqs);
    let pos = input.graph().arange::<Seq>() + prev_seq;
    let emb = pos.expand::<(_, Const<1>), _>().matmul(freqs.expand());

    // Split input into evens and odds
    let split = input.reshape::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>)>();
    let x0: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> = split
        .slice((.., .., .., .., ..Expression::from(1)))
        .realize();
    let x1: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> = split
        .slice((.., .., .., .., Expression::from(1)..))
        .realize();

    // Apply sin and cos embeddings
    let x0_out = x0 * emb.cos().expand() - x1 * emb.sin().expand();
    let x1_out = x0 * emb.sin().expand() + x1 * emb.cos().expand();

    // Combine back into output
    x0_out
        .concat_along::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>), Axis<4>, _>(
            x1_out,
        )
        .reshape()
}

pub struct SelfAttention {
    pub q_proj: PermutedLinear<HIDDEN_DIM, HIDDEN_DIM>,
    pub k_proj: PermutedLinear<HIDDEN_DIM, ATTN_PROJ_DIM>,
    pub v_proj: PermutedLinear<HIDDEN_DIM, ATTN_PROJ_DIM>,
    pub o_proj: PermutedLinear<HIDDEN_DIM, HIDDEN_DIM>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, PrevSeq>,
        PhantomData<TotSeq>,
        usize,
    )> for SelfAttention
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (x, (k_cache, v_cache), _, index): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
            usize,
        ),
    ) -> Self::Output {
        // Apply the Projections
        let queries = self.q_proj.forward(x);
        if index == 0 {
            queries.diff(
                || Some("/Users/jafioti/Desktop/saves/queries.bin".into()),
                1e-8,
            );
        }
        let queries = queries
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let keys = self
            .k_proj
            .forward(x)
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let values = self
            .v_proj
            .forward(x)
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_ggml(queries, PrevSeq::size().into());
        if index == 0 {
            queries.diff(
                || Some("/Users/jafioti/Desktop/saves/query_rope.bin".into()),
                1e-8,
            );
        }
        let keys = apply_rotary_embeddings_ggml(keys, PrevSeq::size().into());

        // Add KV cache
        let keys = k_cache.concat_along::<_, Axis<2>, _>(keys);
        let values = v_cache.concat_along::<_, Axis<2>, _>(values);

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys.permute());
        attention_weights = attention_weights / (HEAD_DIM as f32).sqrt();

        // Causal mask
        let attention_mask = attention_weights.graph().triu::<CurSeq>(1) * f32::NEG_INFINITY;
        attention_weights += attention_mask
            .pad::<(CurSeq, TotSeq)>(((0, 0), (TotSeq::size() - CurSeq::size(), 0)))
            .expand();

        // Calculate final outputs
        let output = attention_weights
            .softmax::<Axis<3>>()
            // Apply distribution to values
            .matmul(values)
            // Merge heads
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, CurSeq, Const<HIDDEN_DIM>)>();
        // Apply output projection
        let output = self.o_proj.forward(output);
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
    }
}

impl InitModule for SelfAttention {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: PermutedLinear::new(false, cx),
            k_proj: PermutedLinear::new(false, cx),
            v_proj: PermutedLinear::new(false, cx),
            o_proj: PermutedLinear::new(false, cx),
        }
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("attn_q", &self.q_proj);
        s.module("attn_v", &self.v_proj);
        s.module("attn_k", &self.k_proj);
        s.module("attn_output", &self.o_proj);
    }
}

pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub attention_norm: LayerNorm<HIDDEN_DIM>,
    pub feed_forward: Mlp<MLP_DIM, HIDDEN_DIM>,
    pub feed_forward_norm: LayerNorm<HIDDEN_DIM>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, PrevSeq>,
        PhantomData<TotSeq>,
        usize,
    )> for TransformerBlock
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (mut x, cache, _, index): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
            usize,
        ),
    ) -> Self::Output {
        // Attention
        let normed = self.attention_norm.forward(x);
        let (y, cache) = self
            .attention
            .forward((normed, cache, PhantomData::<TotSeq>, index));

        // Residual Addition
        x += y;

        // Feed Forward
        let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

        // Residual Addition
        (x + y, cache)
    }
}

impl InitModule for TransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: LayerNorm::new(true, false, false, 1e-5, cx),
            feed_forward: InitModule::initialize(cx),
            feed_forward_norm: LayerNorm::new(true, false, false, 1e-5, cx),
        }
    }
}

impl SerializeModule for TransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("", &self.attention);
        s.module("attn_norm", &self.attention_norm);
        s.module("ffn_norm", &self.feed_forward_norm);
        s.module("", &self.feed_forward);
    }
}

pub struct Phi {
    // Token embeddings
    pub embedding: Embedding<VOCAB_SIZE, HIDDEN_DIM>,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Final Norm layer
    pub norm: LayerNorm<HIDDEN_DIM>,
    // LM Head Layer
    pub lm_head: PermutedLinear<HIDDEN_DIM, VOCAB_SIZE>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq)>,
        &[KVCache<Batch, PrevSeq>],
        PhantomData<TotSeq>,
    )> for Phi
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<VOCAB_SIZE>)>,
        Vec<KVCache<Batch, TotSeq>>,
    );
    fn forward(
        &self,
        (input, cache, _): (
            GraphTensor<(Batch, CurSeq)>,
            &[KVCache<Batch, PrevSeq>],
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Embed tokens
        let mut x = self.embedding.forward(input);
        x.diff(
            || Some("/Users/jafioti/Desktop/saves/embed.bin".into()),
            1e-8,
        );

        // Run through layers and collect new caches
        let mut new_caches = vec![];
        let mut new_cache;
        for (i, layer) in self.layers.iter().enumerate() {
            (x, new_cache) = layer.forward((x, cache[i], PhantomData::<TotSeq>, i));
            new_caches.push(new_cache);
        }
        // Run through last norm and output projection
        let output = self.lm_head.forward(self.norm.forward(x));

        (output, new_caches)
    }
}

impl InitModule for Phi {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            embedding: Embedding {
                weight: cx.named_tensor("Embedding Weight"),
            },
            norm: LayerNorm::new(true, false, false, 1e-5, cx),
            lm_head: PermutedLinear::new(false, cx),
            layers: (0..NUM_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
        }
    }
}

impl SerializeModule for Phi {
    fn serialize(&self, s: &mut Serializer) {
        s.module("token_embd", &self.embedding);
        s.module("output_norm", &self.norm);
        s.module("output", &self.lm_head);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
