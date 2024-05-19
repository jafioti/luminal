use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Conv1D, Embedding, LayerNorm, Linear, PermutedEmbedding, PermutedLinear};
use std::marker::PhantomData;
use std::ops::{Add, Mul};

// Encoder
pub const NUM_MEL_BINS: usize = 80;
pub const D_MODEL: usize = 384;
pub const ENC_LAYERS: usize = 4;
pub const ENC_FFN_DIM: usize = 1536;

// Decoder
pub const VOCAB_SIZE: usize = 51864;
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
    GraphTensor<(Batch, Const<HEADS>, Const<HEAD_DIM>, Seq)>,
    GraphTensor<(Batch, Const<HEADS>, Seq, Const<HEAD_DIM>)>,
);

pub struct SelfAttention<const HIDDEN: usize> {
    pub q_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub q_proj_bias: GraphTensor<R1<HIDDEN>>,
    pub k_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub v_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub v_proj_bias: GraphTensor<R1<HIDDEN>>,
    pub o_proj: GraphTensor<R2<HIDDEN, HIDDEN>>,
    pub o_proj_bias: GraphTensor<R1<HIDDEN>>,
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
        let scale = ((HIDDEN as f32 / HEADS as f32) as f64).powf(-0.25) as f32;
        // Apply the Projections
        let queries = x
            .matmul(self.q_proj.permute())
            .add(self.q_proj_bias.expand())
            .mul(scale)
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let keys = x
            .matmul(self.k_proj.permute())
            .mul(scale)
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 3, 1>>()
            .contiguous();

        let values = x
            .matmul(self.v_proj.permute())
            .add(self.v_proj_bias.expand())
            .reshape::<(Batch, CurSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Add KV cache
        let (keys, values) = if let Some((k_cache, v_cache)) = cache {
            (
                k_cache.concat_along::<_, Axis<3>, _>(keys),
                v_cache.concat_along::<_, Axis<2>, _>(values),
            )
        } else {
            (keys.realize(), values.realize())
        };

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys);

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
            .matmul(self.o_proj.permute())
            .add(self.o_proj_bias.expand());
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous
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
        let scale = ((HIDDEN as f32 / HEADS as f32) as f64).powf(-0.25) as f32;
        // Apply the projections
        let queries = queries
            .matmul(self.q_proj.permute())
            .add(self.q_proj_bias.expand())
            .mul(scale)
            .reshape::<(Batch, DecSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let keys = keys
            .matmul(self.k_proj.permute())
            .mul(scale)
            .reshape::<(Batch, EncSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 3, 1>>()
            .contiguous();
        let values = values
            .matmul(self.v_proj.permute())
            .add(self.v_proj_bias.expand())
            .reshape::<(Batch, EncSeq, Const<HEADS>, Const<HEAD_DIM>)>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys);

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
            .matmul(self.o_proj.permute())
            .add(self.o_proj_bias.expand());

        (output, (keys, values))
    }
}

impl<const HIDDEN: usize> InitModule for SelfAttention<HIDDEN> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj"),
            q_proj_bias: cx.named_tensor("Q Proj Bias"),
            k_proj: cx.named_tensor("K Proj"),
            v_proj: cx.named_tensor("V Proj"),
            v_proj_bias: cx.named_tensor("V Proj Bias"),
            o_proj: cx.named_tensor("O Proj"),
            o_proj_bias: cx.named_tensor("O Proj Bias"),
        }
    }
}

impl<const HIDDEN: usize> SerializeModule for SelfAttention<HIDDEN> {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("q_proj/weight", self.q_proj);
        s.tensor("q_proj/bias", self.q_proj_bias);
        s.tensor("v_proj/weight", self.v_proj);
        s.tensor("v_proj/bias", self.v_proj_bias);
        s.tensor("k_proj/weight", self.k_proj);
        s.tensor("out_proj/weight", self.o_proj);
        s.tensor("out_proj/bias", self.o_proj_bias);
    }
}

pub struct EncoderTransformerBlock {
    pub attention: SelfAttention<D_MODEL>,
    pub attention_norm: LayerNorm<D_MODEL>,
    pub ff1: PermutedLinear<D_MODEL, ENC_FFN_DIM>,
    pub ff1_bias: GraphTensor<R1<ENC_FFN_DIM>>,
    pub ff2: PermutedLinear<ENC_FFN_DIM, D_MODEL>,
    pub ff2_bias: GraphTensor<R1<D_MODEL>>,
    pub feed_forward_norm: LayerNorm<D_MODEL>,
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
        let y = self.ff1.forward(self.feed_forward_norm.forward(x)) + self.ff1_bias.expand();
        let y = self.ff2.forward(y.gelu()) + self.ff2_bias.expand();

        // Residual Addition
        x + y
    }
}

impl InitModule for EncoderTransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: LayerNorm::new(true, true, true, 1e-5, cx),
            ff1: PermutedLinear {
                weight: cx.tensor(),
            },
            ff1_bias: cx.tensor(),
            ff2: PermutedLinear {
                weight: cx.tensor(),
            },
            ff2_bias: cx.tensor(),
            feed_forward_norm: LayerNorm::new(true, true, true, 1e-5, cx),
        }
    }
}

impl SerializeModule for EncoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("self_attn_layer_norm", &self.attention_norm);
        s.module("final_layer_norm", &self.feed_forward_norm);
        s.module("fc1", &self.ff1);
        s.tensor("fc1/bias", self.ff1_bias);
        s.module("fc2", &self.ff2);
        s.tensor("fc2/bias", self.ff2_bias);
    }
}

pub struct AudioEncoder {
    // Conv layers (based on https://github.com/huggingface/candle/blob/59b18d974ec3cad6963b774aa245e23f8c80414f/candle-transformers/src/models/whisper/model.rs#L246)
    pub conv1: Conv1D<N_MEL_BINS, D_MODEL, 3, 1, 0, 1>,
    pub conv2: Conv1D<D_MODEL, D_MODEL, 3, 2, 0, 1>,
    // Transformer layers
    pub layers: Vec<EncoderTransformerBlock>,
    // Post layer norm
    pub post_ln: LayerNorm<D_MODEL>,
}

fn sinusoids<const CHANNELS: usize, Length: Dimension>(
    cx: &mut Graph,
) -> GraphTensor<(Length, Const<CHANNELS>)> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (CHANNELS / 2 - 1) as f32;
    let inv_timescales = (0..CHANNELS / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect::<Vec<_>>();
    let mut inv_timescales = cx
        .tensor::<(Dyn<'-'>,)>()
        .set_dyn(inv_timescales, &[CHANNELS / 2]);
    inv_timescales.shape.dims[0] = (CHANNELS / 2).into();
    let arange = cx.arange::<Length>();
    let mut mul_shape = arange.shape;
    mul_shape.add_dim(1, CHANNELS / 2);
    let scaled_time: GraphTensor<(Length, Dyn<'-'>)> =
        arange.expand_to(mul_shape) * inv_timescales.expand_to(mul_shape);
    scaled_time
        .sin()
        .concat_along::<_, Axis<1>, _>(scaled_time.cos())
}

impl<Batch: Dimension, Seq: Dimension, SeqDivTwo: Dimension>
    Module<(
        GraphTensor<(Batch, Const<N_MEL_BINS>, Seq)>,
        PhantomData<SeqDivTwo>,
    )> for AudioEncoder
{
    type Output = GraphTensor<(Batch, SeqDivTwo, Const<D_MODEL>)>;
    fn forward(
        &self,
        (x, _): (
            GraphTensor<(Batch, Const<N_MEL_BINS>, Seq)>,
            PhantomData<SeqDivTwo>,
        ),
    ) -> Self::Output {
        // Conv layers
        let x = self.conv1.forward((x, PhantomData::<Seq>)).gelu();
        let x = self.conv2.forward((x, PhantomData::<SeqDivTwo>)).gelu();
        let mut x = x.permute::<_, Axes3<0, 2, 1>>();
        // Sinusoidal positional embedding
        x += sinusoids::<D_MODEL, SeqDivTwo>(x.graph()).expand();

        // Transformer layers
        let out = self.layers.forward(x);
        // Final norm
        self.post_ln.forward(out)
    }
}

impl InitModule for AudioEncoder {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            conv1: Conv1D::initialize_bias(cx),
            conv2: Conv1D::initialize_bias(cx),
            layers: (0..ENC_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
            post_ln: LayerNorm::new(true, true, true, 1e-5, cx),
        }
    }
}

impl SerializeModule for AudioEncoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/encoder/conv1", &self.conv1);
        s.module("model/encoder/conv2", &self.conv2);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("model/encoder/layers/{i}"), layer);
        }
        s.module("model/encoder/layer_norm", &self.post_ln);
    }
}

pub struct DecoderTransformerBlock {
    pub attention: SelfAttention<D_MODEL>,
    pub attention_norm: LayerNorm<D_MODEL>,
    pub cross_attention: SelfAttention<D_MODEL>,
    pub cross_attention_norm: LayerNorm<D_MODEL>,
    pub ff1: PermutedLinear<D_MODEL, DEC_FFN_DIM>,
    pub ff1_bias: GraphTensor<R1<DEC_FFN_DIM>>,
    pub ff2: PermutedLinear<DEC_FFN_DIM, D_MODEL>,
    pub ff2_bias: GraphTensor<R1<D_MODEL>>,
    pub feed_forward_norm: LayerNorm<D_MODEL>,
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
        (mut x, encoded, cache, _): (
            GraphTensor<(Batch, CurSeq, Const<D_MODEL>)>,
            GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>,
            KVCache<Batch, PrevSeq>,
            PhantomData<TotSeq>,
        ),
    ) -> Self::Output {
        // Self Attention
        let (y, cache) = self.attention.forward((
            self.attention_norm.forward(x),
            Some(cache),
            true,
            PhantomData::<TotSeq>,
        ));

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
        let y = self.ff1.forward(self.feed_forward_norm.forward(x)) + self.ff1_bias.expand();
        let y = self.ff2.forward(y.gelu()) + self.ff2_bias.expand();

        // Residual Addition
        (x + y, enc_states, cache)
    }
}

impl InitModule for DecoderTransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: LayerNorm::new(true, true, true, 1e-5, cx),
            cross_attention: InitModule::initialize(cx),
            cross_attention_norm: LayerNorm::new(true, true, true, 1e-5, cx),
            ff1: PermutedLinear {
                weight: cx.tensor(),
            },
            ff1_bias: cx.tensor(),
            ff2: PermutedLinear {
                weight: cx.tensor(),
            },
            ff2_bias: cx.tensor(),
            feed_forward_norm: LayerNorm::new(true, true, true, 1e-5, cx),
        }
    }
}

impl SerializeModule for DecoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("self_attn_layer_norm", &self.attention_norm);
        s.module("encoder_attn", &self.cross_attention);
        s.module("encoder_attn_layer_norm", &self.cross_attention_norm);
        s.module("fc1", &self.ff1);
        s.tensor("fc1/bias", self.ff1_bias);
        s.module("fc2", &self.ff2);
        s.tensor("fc2/bias", self.ff2_bias);
        s.module("final_layer_norm", &self.feed_forward_norm);
    }
}

pub struct TextDecoder {
    // Embeddings
    pub embedding: PermutedEmbedding<VOCAB_SIZE, D_MODEL>,
    pub pos_embedding: GraphTensor<R2<MAX_TARGET_POSITION, D_MODEL>>,
    // Transformer layers
    pub layers: Vec<DecoderTransformerBlock>,
    // Final layer norm
    pub layer_norm: LayerNorm<D_MODEL>,
}

impl<
        Batch: Dimension,
        EncSeq: Dimension,
        PrevDecSeq: Dimension,
        CurDecSeq: Dimension,
        TotDecSeq: Dimension,
    >
    Module<(
        GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>,
        GraphTensor<(Batch, CurDecSeq)>,
        &[KVCache<Batch, PrevDecSeq>],
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
            GraphTensor<(Batch, EncSeq, Const<D_MODEL>)>,
            GraphTensor<(Batch, CurDecSeq)>,
            &[KVCache<Batch, PrevDecSeq>],
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
            .contiguous()
            .realize::<(CurDecSeq, Const<D_MODEL>)>()
            .expand();
        // Run through layers and collect new caches
        let (mut new_caches, mut enc_states) = (vec![], vec![]);
        let (mut new_cache, mut enc_state);
        for (i, layer) in self.layers.iter().enumerate() {
            (x, enc_state, new_cache) =
                layer.forward((x, enc_output, cache[i], PhantomData::<TotDecSeq>));
            new_caches.push(new_cache);
            enc_states.push(enc_state);
        }
        // Run through last norm and output projection
        (
            self.layer_norm.forward(x).matmul(
                self.embedding
                    .weight
                    .realize::<R2<VOCAB_SIZE, D_MODEL>>()
                    .permute(),
            ),
            enc_states,
            new_caches,
        )
    }
}

impl InitModule for TextDecoder {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            embedding: InitModule::initialize(cx),
            pos_embedding: cx.tensor(),
            layer_norm: LayerNorm::new(true, true, true, 1e-5, cx),
            layers: (0..DEC_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
        }
    }
}

impl SerializeModule for TextDecoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/decoder/embed_tokens", &self.embedding);
        s.tensor("model/decoder/embed_positions/weight", self.pos_embedding);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("model/decoder/layers/{i}"), layer);
        }
        s.module("model/decoder/layer_norm", &self.layer_norm);
    }
}
