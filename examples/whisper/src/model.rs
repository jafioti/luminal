use std::marker::PhantomData;

use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Conv1D, Embedding, RMSNorm};

// Encoder
pub const N_MEL_BINS: usize = 1;
pub const N_STATE: usize = 1;

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
pub const N_HEADS: usize = 32;
pub const MLP_DIM: usize = 14336;

pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS;
pub const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
pub const ATTN_PROJ_DIM: usize = HEAD_DIM * N_HEADS;

pub type KVCache<Batch, Seq> = (
    GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
    GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
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
        s.tensor("ffn_gate/weight", self.gate_proj);
        s.tensor("ffn_up/weight", self.up_proj);
        s.tensor("ffn_down/weight", self.down_proj);
    }
}

fn apply_rotary_embeddings_ggml<const N_HEADS: usize, Batch: Dimension, Seq: Dimension>(
    input: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)>,
    prev_seq: BigExpression,
) -> GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM>)> {
    // Get freqs
    let freqs = (input.graph().arange::<Const<HEAD_DIM_OVER_2>>() * 2.0) / (HEAD_DIM as f32);
    let freqs = 1000000_f32.pow(freqs);
    let pos = input.graph().arange::<Seq>() + prev_seq;
    let emb = pos.expand::<(_, Const<1>), _>().matmul(freqs.expand());

    // Split input into evens and odds
    let split = input.reshape::<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<2>)>();
    let x0: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> = split
        .slice((.., .., .., .., ..Expression::from(1)))
        .contiguous()
        .realize();
    let x1: GraphTensor<(Batch, Const<N_HEADS>, Seq, Const<HEAD_DIM_OVER_2>, Const<1>)> = split
        .slice((.., .., .., .., Expression::from(1)..))
        .contiguous()
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

pub struct SelfAttention<const HIDDEN: usize> {
    pub q_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub k_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub v_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub o_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
}

impl<
        const HIDDEN: usize,
        Batch: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        Option<KVCache<Batch, PrevSeq>>,
        bool,
        PhantomData<TotSeq>,
    )> for SelfAttention<HIDDEN>
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (x, cache, mask, _): (
            GraphTensor<(Batch, CurSeq, Const<HIDDEN>)>,
            Option<KVCache<Batch, PrevSeq>>,
            bool,
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
        let (keys, values) = if let Some((k_cache, v_cache)) = cache {
            (
                k_cache.concat_along::<_, Axis<2>, _>(keys),
                v_cache.concat_along::<_, Axis<2>, _>(values),
            )
        } else {
            (keys.realize(), values.realize())
        };

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys.permute()) / (HEAD_DIM as f32).sqrt();

        if mask {
            let attention_mask = self.k_proj.graph().triu::<CurSeq>(1) * f16::MIN.to_f32();
            attention_weights += attention_mask
                .pad::<(CurSeq, TotSeq)>(((0, 0), (TotSeq::size() - CurSeq::size(), 0)))
                .expand();
        }

        // Calculate final outputs
        let output = attention_weights
            .softmax::<Axis<3>>()
            // Apply distribution to values
            .matmul(values)
            // Merge heads
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, CurSeq, Const<HIDDEN>)>();
        let output = output
            // Apply output projection
            .matmul(self.o_proj.permute());
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
    }
}

impl<const HIDDEN: usize> SelfAttention<HIDDEN> {
    #[allow(clippy::type_complexity)]
    fn cross_attention_forward<Batch: Dimension, EncSeq: Dimension, DecSeq: Dimension>(
        &self,
        queries: GraphTensor<(Batch, DecSeq, Const<HIDDEN>)>,
        keys: GraphTensor<(Batch, EncSeq, Const<HIDDEN>)>,
        values: GraphTensor<(Batch, EncSeq, Const<HIDDEN>)>,
    ) -> (
        GraphTensor<(Batch, DecSeq, Const<HIDDEN>)>,
        GraphTensor<(Batch, Const<N_HEADS>, EncSeq, Const<HEAD_DIM>)>,
        GraphTensor<(Batch, Const<N_HEADS>, EncSeq, Const<HEAD_DIM>)>,
    ) {
        // Apply the projections
        let queries = queries
            .matmul(self.q_proj.permute())
            .reshape::<(Batch, DecSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let keys = keys
            .matmul(self.k_proj.permute())
            .reshape::<(Batch, EncSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let values = values
            .matmul(self.v_proj.permute())
            .reshape::<(Batch, EncSeq, Const<N_HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys.permute()) / (HEAD_DIM as f32).sqrt();

        let attention_mask = self.k_proj.graph().triu::<DecSeq>(1) * f16::MIN.to_f32();
        attention_weights += attention_mask
            .pad::<(DecSeq, EncSeq)>(((0, 0), (EncSeq::size() - DecSeq::size(), 0)))
            .expand();

        // Calculate final outputs
        let output = attention_weights
            .softmax::<Axis<3>>()
            // Apply distribution to values
            .matmul(values)
            // Merge heads
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, DecSeq, Const<HIDDEN>)>();
        let output = output
            // Apply output projection
            .matmul(self.o_proj.permute());

        (output, keys, values)
    }
}

impl<const HIDDEN: usize> InitModule for SelfAttention<HIDDEN> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj"),
            k_proj: cx.named_tensor("K Proj"),
            v_proj: cx.named_tensor("V Proj"),
            o_proj: cx.named_tensor("O Proj"),
        }
    }
}

impl<const HIDDEN: usize> SerializeModule for SelfAttention<HIDDEN> {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("attn_q/weight", self.q_proj);
        s.tensor("attn_v/weight", self.v_proj);
        s.tensor("attn_k/weight", self.k_proj);
        s.tensor("attn_output/weight", self.o_proj);
    }
}

pub struct EncoderTransformerBlock<const HIDDEN: usize> {
    pub attention: SelfAttention<HIDDEN>,
    pub attention_norm: RMSNorm<HIDDEN>,
    pub feed_forward: Mlp<MLP_DIM, HIDDEN>,
    pub feed_forward_norm: RMSNorm<HIDDEN>,
}

impl<const HIDDEN: usize, Batch: Dimension, Seq: Dimension>
    Module<GraphTensor<(Batch, Seq, Const<HIDDEN>)>> for EncoderTransformerBlock<HIDDEN>
{
    type Output = GraphTensor<(Batch, Seq, Const<HIDDEN>)>;
    fn forward(&self, mut x: GraphTensor<(Batch, Seq, Const<HIDDEN>)>) -> Self::Output {
        // Attention
        let (y, _) = self.attention.forward((
            self.attention_norm.forward(x),
            Option::<KVCache<Batch, Seq>>::None,
            false,
            PhantomData::<Seq>,
        ));

        // Residual Addition
        x += y;

        // Feed Forward
        let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

        // Residual Addition
        x + y
    }
}

impl<const HIDDEN: usize> InitModule for EncoderTransformerBlock<HIDDEN> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: {
                let mut norm = RMSNorm::initialize(cx);
                norm.epsilon = 1e-5;
                norm
            },
            feed_forward: InitModule::initialize(cx),
            feed_forward_norm: {
                let mut norm = RMSNorm::initialize(cx);
                norm.epsilon = 1e-5;
                norm
            },
        }
    }
}

impl<const HIDDEN: usize> SerializeModule for EncoderTransformerBlock<HIDDEN> {
    fn serialize(&self, s: &mut Serializer) {
        s.module("", &self.attention);
        s.module("attn_norm", &self.attention_norm);
        s.module("ffn_norm", &self.feed_forward_norm);
        s.module("", &self.feed_forward);
    }
}

pub struct AudioEncoder {
    // Conv layers (based on https://github.com/huggingface/candle/blob/59b18d974ec3cad6963b774aa245e23f8c80414f/candle-transformers/src/models/whisper/model.rs#L246)
    pub conv1: Conv1D<N_MEL_BINS, N_STATE, 3, 1, 1>,
    pub conv2: Conv1D<N_STATE, N_STATE, 3, 2, 1>,
    // Transformer layers
    pub layers: Vec<EncoderTransformerBlock<N_STATE>>,
}

fn sinusoids<const CHANNELS: usize, Length: Dimension>(
    cx: &mut Graph,
) -> GraphTensor<(Length, Const<CHANNELS>)> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (CHANNELS / 2 - 1) as f32;
    let inv_timescales = (0..CHANNELS / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect::<Vec<_>>();
    let inv_timescales = cx
        .tensor::<(Dyn<'-'>,)>()
        .set_dyn(inv_timescales, &[CHANNELS / 2]);
    let arange = cx.arange::<Length>();
    let scaled_time: GraphTensor<(Length, Dyn<'-'>)> = arange.expand() * inv_timescales.expand();
    scaled_time
        .sin()
        .concat_along::<_, Axis<1>, _>(scaled_time.cos())
}

impl<Batch: Dimension, Seq: Dimension> Module<GraphTensor<(Batch, Seq, Const<N_MEL_BINS>)>>
    for AudioEncoder
{
    type Output = GraphTensor<(Batch, Seq, Const<N_STATE>)>;
    fn forward(&self, input: GraphTensor<(Batch, Seq, Const<N_MEL_BINS>)>) -> Self::Output {
        // Conv layers
        let x = self
            .conv1
            .forward((input.permute::<_, Axes3<0, 2, 1>>(), PhantomData::<Seq>))
            .gelu();
        let mut x = self
            .conv2
            .forward((x, PhantomData::<Seq>))
            .gelu()
            .permute::<_, Axes3<0, 2, 1>>();
        // Sinusoidal positional embedding
        x += sinusoids::<N_MEL_BINS, Seq>(x.graph()).expand();
        // Transformer layers
        let output = self.layers.forward(x);
        // Final layer norm
        output.layer_norm::<Axis<2>, _>(1e-5)
    }
}

impl InitModule for AudioEncoder {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            conv1: InitModule::initialize(cx),
            conv2: InitModule::initialize(cx),
            layers: (0..NUM_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
        }
    }
}

impl SerializeModule for AudioEncoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("conv1", &self.conv1);
        s.module("conv2", &self.conv2);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
