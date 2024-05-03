use std::marker::PhantomData;

use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Embedding, PermutedLinear, RMSNorm};

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

impl<Sh: Shape, Im: Shape, const I: usize, const H: usize> Module<GraphTensor<Sh>> for Mlp<I, H>
where
    GraphTensor<Sh>: Matmul<R2<H, I>, Output = GraphTensor<Im>>,
    GraphTensor<Im>: Matmul<R2<I, H>, Output = GraphTensor<Sh>>,
{
    type Output = GraphTensor<Sh>;

    fn forward(&self, input: GraphTensor<Sh>) -> Self::Output {
        let gate = self.gate_proj.forward(input).swish();
        let up = self.up_proj.forward(input) * gate;
        self.down_proj.forward(up)
    }
}

impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            gate_proj: PermutedLinear {
                weight: cx.named_tensor("Gate"),
            },
            up_proj: PermutedLinear {
                weight: cx.named_tensor("Up"),
            },
            down_proj: PermutedLinear {
                weight: cx.named_tensor("Down"),
            },
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
    let freqs = 500000_f32.pow(freqs);
    let pos = input.graph().arange::<Seq>() + prev_seq;
    let emb = pos.expand::<(_, Const<1>), _>().matmul(freqs.expand());

    // Split input into evens and odds
    let split = input.reshape::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>)>();
    let x0: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> =
        split.slice((.., .., .., .., ..1)).realize();
    let x1: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> =
        split.slice((.., .., .., .., 1..)).realize();

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
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub k_proj: GraphTensor<R2<ATTN_PROJ_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<ATTN_PROJ_DIM, HIDDEN_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, PrevSeq>,
        PhantomData<TotSeq>,
    )> for SelfAttention
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (x, (k_cache, v_cache), _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Apply the Projections
        let queries = x
            .matmul(self.q_proj.permute())
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let keys = x
            .matmul(self.k_proj.permute())
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let values = x
            .matmul(self.v_proj.permute())
            .reshape::<(Batch, CurSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_ggml(queries, PrevSeq::size().into());
        let keys = apply_rotary_embeddings_ggml(keys, PrevSeq::size().into());

        // Add KV cache
        let keys = k_cache.concat_along::<_, Axis<2>, _>(keys);
        let values = v_cache.concat_along::<_, Axis<2>, _>(values);

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys.permute()) / (HEAD_DIM as f32).sqrt();

        let attention_mask = self.k_proj.graph().triu::<CurSeq>(1) * f16::MIN.to_f32();
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
        let output = output
            // Apply output projection
            .matmul(self.o_proj.permute());
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
    }
}

impl InitModule for SelfAttention {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj"),
            k_proj: cx.named_tensor("K Proj"),
            v_proj: cx.named_tensor("V Proj"),
            o_proj: cx.named_tensor("O Proj"),
        }
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("attn_q/weight", self.q_proj);
        s.tensor("attn_v/weight", self.v_proj);
        s.tensor("attn_k/weight", self.k_proj);
        s.tensor("attn_output/weight", self.o_proj);
    }
}

pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub attention_norm: RMSNorm<HIDDEN_DIM>,
    pub feed_forward: Mlp<MLP_DIM, HIDDEN_DIM>,
    pub feed_forward_norm: RMSNorm<HIDDEN_DIM>,
}

impl<Batch: Dimension, CurSeq: Dimension, PrevSeq: Dimension, TotSeq: Dimension>
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, PrevSeq>,
        PhantomData<TotSeq>,
    )> for TransformerBlock
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (mut x, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN_DIM>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Attention
        let normed = self.attention_norm.forward(x);
        let (y, cache) = self
            .attention
            .forward((normed, cache, PhantomData::<TotSeq>));

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
            attention_norm: RMSNorm {
                weight: cx.named_tensor("RMS Norm Weight"),
                epsilon: 1e-5,
            },
            feed_forward: InitModule::initialize(cx),
            feed_forward_norm: RMSNorm {
                weight: cx.named_tensor("RMS Norm Weight"),
                epsilon: 1e-5,
            },
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
    pub norm: RMSNorm<HIDDEN_DIM>,
    // LM Head Layer
    pub lm_head: GraphTensor<R2<VOCAB_SIZE, HIDDEN_DIM>>,
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

        // Run through layers and collect new caches
        let mut new_caches = vec![];
        let mut new_cache;
        for (i, layer) in self.layers.iter().enumerate() {
            (x, new_cache) = layer.forward((x, cache[i], PhantomData::<TotSeq>));
            new_caches.push(new_cache);
        }
        // Run through last norm and output projection
        let output = self.norm.forward(x).matmul(self.lm_head.permute());

        (output, new_caches)
    }
}

impl InitModule for Phi {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            embedding: Embedding {
                weight: cx.named_tensor("Embedding Weight"),
            },
            norm: RMSNorm {
                weight: cx.named_tensor("RMS Norm Weight"),
                epsilon: 1e-5,
            },
            lm_head: cx.named_tensor("LM Head"),
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
        s.tensor("output/weight", self.lm_head);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
