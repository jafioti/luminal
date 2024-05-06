use std::marker::PhantomData;

use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Conv1D, Embedding, PermutedLinear, RMSNorm};

// Encoder
pub const NUM_MEL_BINS: usize = 80;
pub const D_MODEL: usize = 384;
pub const ENC_LAYERS: usize = 4;
pub const ENC_FFN_DIM: usize = 1536;

// Decoder
pub const VOCAB_SIZE: usize = 51865;
pub const DEC_LAYERS: usize = 4;
pub const DEC_FFN_DIM: usize = 1536;

// Shared
pub const HEADS: usize = 6;
pub const HEAD_DIM: usize = D_MODEL / HEADS;
pub const MAX_TARGET_POSITION: usize = 448;

// Audio parameters.
pub const N_MEL_BINS: usize = 80;
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples in a 30-second chunk
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 frames in a mel spectrogram input

pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

// Tokenizer dependent bits.
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

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
        s.tensor("ffn_gate/weight", self.gate_proj);
        s.tensor("ffn_up/weight", self.up_proj);
        s.tensor("ffn_down/weight", self.down_proj);
    }
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
        KVCache<Batch, EncSeq>,
    ) {
        // Apply the projections
        let queries = queries
            .matmul(self.q_proj.permute())
            .reshape::<(Batch, DecSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let keys = keys
            .matmul(self.k_proj.permute())
            .reshape::<(Batch, EncSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let values = values
            .matmul(self.v_proj.permute())
            .reshape::<(Batch, EncSeq, Const<HEADS>, Const<HEAD_DIM>)>()
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

        (output, (keys, values))
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

pub struct EncoderTransformerBlock {
    pub attention: SelfAttention<D_MODEL>,
    pub attention_norm: RMSNorm<D_MODEL>,
    pub feed_forward: Mlp<ENC_FFN_DIM, D_MODEL>,
    pub feed_forward_norm: RMSNorm<D_MODEL>,
}

impl<Batch: Dimension, Seq: Dimension> Module<GraphTensor<(Batch, Seq, Const<D_MODEL>)>>
    for EncoderTransformerBlock
{
    type Output = GraphTensor<(Batch, Seq, Const<D_MODEL>)>;
    fn forward(&self, mut x: GraphTensor<(Batch, Seq, Const<D_MODEL>)>) -> Self::Output {
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

impl InitModule for EncoderTransformerBlock {
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

impl SerializeModule for EncoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("", &self.attention);
        s.module("attn_norm", &self.attention_norm);
        s.module("ffn_norm", &self.feed_forward_norm);
        s.module("", &self.feed_forward);
    }
}

pub struct AudioEncoder {
    // Conv layers (based on https://github.com/huggingface/candle/blob/59b18d974ec3cad6963b774aa245e23f8c80414f/candle-transformers/src/models/whisper/model.rs#L246)
    pub conv1: Conv1D<N_MEL_BINS, D_MODEL, 3, 1, 1>,
    pub conv2: Conv1D<D_MODEL, D_MODEL, 3, 2, 1>,
    // Transformer layers
    pub layers: Vec<EncoderTransformerBlock>,
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
    type Output = GraphTensor<(Batch, Seq, Const<D_MODEL>)>;
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
        x += sinusoids::<D_MODEL, Seq>(x.graph()).expand();
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
            layers: (0..ENC_LAYERS)
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

pub struct DecoderTransformerBlock {
    pub attention: SelfAttention<D_MODEL>,
    pub attention_norm: RMSNorm<D_MODEL>,
    pub cross_attention: SelfAttention<D_MODEL>,
    pub cross_attention_norm: RMSNorm<D_MODEL>,
    pub feed_forward: Mlp<DEC_FFN_DIM, D_MODEL>,
    pub feed_forward_norm: RMSNorm<D_MODEL>,
}

impl<
        Batch: Dimension,
        EncSeq: Dimension,
        CurSeq: Dimension,
        PrevSeq: Dimension,
        TotSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, CurSeq, Const<D_MODEL>)>,
        GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>,
        KVCache<Batch, PrevSeq>,
        PhantomData<TotSeq>,
    )> for DecoderTransformerBlock
{
    type Output = (
        GraphTensor<(Batch, CurSeq, Const<D_MODEL>)>,
        KVCache<Batch, EncSeq>,
        KVCache<Batch, TotSeq>,
    );
    fn forward(
        &self,
        (mut x, encoded, cache, tot): (
            GraphTensor<(Batch, CurSeq, Const<D_MODEL>)>,
            GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Self Attention
        let (y, cache) =
            self.attention
                .forward((self.attention_norm.forward(x), Some(cache), true, tot));

        // Residual Addition
        x += y;

        // Cross Attention
        let (y, enc_states) = self.cross_attention.cross_attention_forward(
            self.cross_attention_norm.forward(x),
            encoded,
            encoded,
        );

        // Residual Addition
        x += y;

        // Feed Forward
        let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

        // Residual Addition
        (x + y, enc_states, cache)
    }
}

impl InitModule for DecoderTransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: {
                let mut norm = RMSNorm::initialize(cx);
                norm.epsilon = 1e-5;
                norm
            },
            cross_attention: InitModule::initialize(cx),
            cross_attention_norm: {
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

impl SerializeModule for DecoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("", &self.attention);
        s.module("attn_norm", &self.attention_norm);
        s.module("", &self.cross_attention);
        s.module("attn_norm", &self.cross_attention);
        s.module("ffn_norm", &self.feed_forward_norm);
        s.module("", &self.feed_forward);
    }
}

pub struct TextDecoder {
    // Embeddings
    pub embedding: Embedding<VOCAB_SIZE, D_MODEL>,
    pub pos_embedding: GraphTensor<R2<MAX_TARGET_POSITION, D_MODEL>>,
    // Transformer layers
    pub layers: Vec<DecoderTransformerBlock>,
    // LM head
    pub lm_head: (RMSNorm<D_MODEL>, PermutedLinear<D_MODEL, VOCAB_SIZE>),
}

impl<
        Batch: Dimension,
        EncSeq: Dimension,
        PrevDecSeq: Dimension,
        CurDecSeq: Dimension,
        TotDecSeq: Dimension,
    >
    Module<(
        Vec<GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>>,
        GraphTensor<(Batch, CurDecSeq)>,
        Vec<KVCache<Batch, PrevDecSeq>>,
        PhantomData<TotDecSeq>,
    )> for TextDecoder
{
    type Output = (
        GraphTensor<(Batch, CurDecSeq, Const<VOCAB_SIZE>)>,
        Vec<KVCache<Batch, EncSeq>>,    // Encoder projected states
        Vec<KVCache<Batch, TotDecSeq>>, // Decoder KV cache
    );
    fn forward(
        &self,
        (enc_output, input, cache, _): (
            Vec<GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>>,
            GraphTensor<(Batch, CurDecSeq)>,
            Vec<KVCache<Batch, PrevDecSeq>>,
            PhantomData<TotDecSeq>,
        ),
    ) -> Self::Output {
        // Embed text
        let mut x = self.embedding.forward(input);
        x += self
            .pos_embedding
            .slice((
                PrevDecSeq::size()..CurDecSeq::size() + PrevDecSeq::size(),
                ..,
            ))
            .realize::<(CurDecSeq, Const<D_MODEL>)>()
            .expand();
        // Run through layers and collect new caches
        let (mut new_caches, mut enc_states) = (vec![], vec![]);
        let (mut new_cache, mut enc_state);
        for (i, layer) in self.layers.iter().enumerate() {
            (x, enc_state, new_cache) =
                layer.forward((x, enc_output[i], cache[i], PhantomData::<TotDecSeq>));
            new_caches.push(new_cache);
            enc_states.push(enc_state);
        }
        // Run through last norm and output projection
        (self.lm_head.forward(x), enc_states, new_caches)
    }
}

impl InitModule for TextDecoder {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            embedding: InitModule::initialize(cx),
            pos_embedding: cx.tensor(),
            lm_head: InitModule::initialize(cx),
            layers: (0..DEC_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
        }
    }
}

impl SerializeModule for TextDecoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("embedding", &self.embedding);
        s.module("head", &self.lm_head);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
