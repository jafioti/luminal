use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Conv1D, Embedding, GeLU, LayerNorm, Linear};
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

pub type KVCache = (GraphTensor, GraphTensor);

pub struct SelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
}

impl Module<(GraphTensor, Option<KVCache>, bool)> for SelfAttention {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, cache, mask): (GraphTensor, Option<KVCache>, bool)) -> Self::Output {
        // x: batch, seq, hidden
        let (batch, seq, hidden) = x.dims3();
        let scale = ((hidden.to_usize().unwrap() as f32 / HEADS as f32) as f64).powf(-0.25) as f32;
        // Apply the Projections
        let queries = self
            .q_proj
            .forward(x)
            .mul(scale)
            .reshape((batch, seq, HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        let mut keys = self
            .k_proj
            .forward(x)
            .mul(scale)
            .reshape((batch, seq, HEADS, HEAD_DIM))
            .permute((0, 2, 3, 1))
            .contiguous();

        let mut values = self
            .v_proj
            .forward(x)
            .reshape((batch, seq, HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        // Add KV cache
        if let Some((k_cache, v_cache)) = cache {
            keys = k_cache.concat_along(keys, 3);
            values = v_cache.concat_along(values, 2);
        }

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys);

        if mask {
            let mut attention_mask = queries.graph().triu(seq, 1) * f16::MIN.to_f32();
            if let Some((c, _)) = cache {
                let (_, _, _, prev_seq) = c.dims4();
                attention_mask = attention_mask.pad(((0, 0), (prev_seq, 0)));
            }
            attention_weights += attention_mask.expand_dim(0, batch).expand_dim(1, HEADS);
        }

        // Calculate final outputs
        let output = attention_weights
            .softmax(3)
            // Apply distribution to values
            .matmul(values)
            // Merge heads
            .permute((0, 2, 1, 3))
            .reshape((batch, seq, hidden));
        // Apply output projection
        (
            self.o_proj.forward(output),
            (keys.contiguous(), values.contiguous()), // Cache needs to be contiguous
        )
    }
}

impl SelfAttention {
    fn cross_attention_forward(
        &self,
        queries: GraphTensor, // batch, dec_seq, hidden
        keys: GraphTensor,    // batch, enc_seq, hidden
        values: GraphTensor,  // batch, enc_seq, hidden
    ) -> (
        GraphTensor, // batch, dec_seq, hidden
        KVCache,
    ) {
        let (batch, enc_seq, hidden) = keys.dims3();
        let (_, dec_seq, _) = queries.dims3();
        let scale = ((hidden.to_usize().unwrap() as f32 / HEADS as f32) as f64).powf(-0.25) as f32;
        // Apply the projections
        let queries = self
            .q_proj
            .forward(queries)
            .mul(scale)
            .reshape((batch, dec_seq, HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));
        let keys = self
            .k_proj
            .forward(keys)
            .mul(scale)
            .reshape((batch, enc_seq, HEADS, HEAD_DIM))
            .permute((0, 2, 3, 1))
            .contiguous();
        let values = self
            .v_proj
            .forward(values)
            .reshape((batch, enc_seq, HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        // Calculate attention weights
        let mut attention_weights = queries.matmul(keys);

        // Calculate final outputs
        let output = attention_weights
            .softmax(3)
            // Apply distribution to values
            .matmul(values)
            // Merge heads
            .permute((0, 2, 1, 3))
            .reshape((batch, dec_seq, hidden));

        // Apply output projection
        (self.o_proj.forward(output), (keys, values))
    }
}

impl SelfAttention {
    fn new(hidden: usize, cx: &mut Graph) -> Self {
        Self {
            q_proj: Linear::new_permuted(hidden, hidden, true, cx),
            k_proj: Linear::new_permuted(hidden, hidden, false, cx),
            v_proj: Linear::new_permuted(hidden, hidden, true, cx),
            o_proj: Linear::new_permuted(hidden, hidden, true, cx),
        }
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("q_proj", &self.q_proj);
        s.module("k_proj", &self.k_proj);
        s.module("v_proj", &self.v_proj);
        s.module("out_proj", &self.o_proj);
    }
}

pub struct EncoderTransformerBlock {
    pub attn: SelfAttention,
    pub attn_norm: LayerNorm,
    pub ff: (Linear, GeLU, Linear),
    pub feed_forward_norm: LayerNorm,
}

impl Module<GraphTensor> for EncoderTransformerBlock {
    type Output = GraphTensor;
    fn forward(&self, mut x: GraphTensor) -> Self::Output {
        let (batch, seq, _) = x.dims3();
        // Attention
        let (y, _) = self.attn.forward((self.attn_norm.forward(x), None, false));

        // Residual Addition
        x += y;

        // Feed Forward
        let y = self.ff.forward(self.feed_forward_norm.forward(x));

        // Residual Addition
        x + y
    }
}

impl EncoderTransformerBlock {
    fn new(hidden: usize, ff: usize, cx: &mut Graph) -> Self {
        Self {
            attn: SelfAttention::new(hidden, cx),
            attn_norm: LayerNorm::new(hidden, true, true, true, 1e-5, cx),
            ff: (
                Linear::new_permuted(hidden, ff, true, cx),
                GeLU,
                Linear::new_permuted(ff, hidden, true, cx),
            ),
            feed_forward_norm: LayerNorm::new(hidden, true, true, true, 1e-5, cx),
        }
    }
}

impl SerializeModule for EncoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attn);
        s.module("self_attn_layer_norm", &self.attn_norm);
        s.module("final_layer_norm", &self.feed_forward_norm);
        s.module("fc1", &self.ff.0);
        s.module("fc2", &self.ff.2);
    }
}

pub struct AudioEncoder {
    // Conv layers (based on https://github.com/huggingface/candle/blob/59b18d974ec3cad6963b774aa245e23f8c80414f/candle-transformers/src/models/whisper/model.rs#L246)
    pub conv1: Conv1D,
    pub conv2: Conv1D,
    // Transformer layers
    pub layers: Vec<EncoderTransformerBlock>,
    // Post layer norm
    pub post_ln: LayerNorm,
}

fn sinusoids(channels: usize, length: Expression, cx: &mut Graph) -> GraphTensor {
    let max_timescale = 10_000_f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect::<Vec<_>>();
    let mut inv_timescales = cx.tensor(channels / 2).set(inv_timescales);
    let scaled_time =
        cx.arange(length).expand_dim(1, channels / 2) * inv_timescales.expand_dim(0, length);
    scaled_time.sin().concat_along(scaled_time.cos(), 1)
}

impl Module<GraphTensor> for AudioEncoder {
    type Output = GraphTensor;
    fn forward(&self, x: GraphTensor) -> Self::Output {
        let (_, _, seq) = x.dims3();

        // Conv layers
        let x = self.conv1.forward(x).gelu();
        let x = self.conv2.forward(x).gelu();
        let mut x = x.permute((0, 2, 1));

        // Sinusoidal positional embedding
        x += sinusoids(D_MODEL, seq / 2, x.graph()).expand_to(x.shape);

        // Transformer layers
        let out = self.layers.forward(x);

        // Final norm
        self.post_ln.forward(out)
    }
}

impl AudioEncoder {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            conv1: Conv1D::new(N_MEL_BINS, D_MODEL, 3, 1, 1, 1, true, cx),
            conv2: Conv1D::new(D_MODEL, D_MODEL, 3, 2, 1, 1, true, cx),
            layers: (0..ENC_LAYERS)
                .map(|_| EncoderTransformerBlock::new(D_MODEL, ENC_FFN_DIM, cx))
                .collect(),
            post_ln: LayerNorm::new(D_MODEL, true, true, true, 1e-5, cx),
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
    pub attention: SelfAttention,
    pub attention_norm: LayerNorm,
    pub cross_attention: SelfAttention,
    pub cross_attention_norm: LayerNorm,
    pub ff: (Linear, GeLU, Linear),
    pub feed_forward_norm: LayerNorm,
}

impl Module<(GraphTensor, GraphTensor, KVCache)> for DecoderTransformerBlock {
    type Output = (GraphTensor, KVCache, KVCache);
    fn forward(
        &self,
        (mut x, encoded, cache): (GraphTensor, GraphTensor, KVCache),
    ) -> Self::Output {
        // Self Attention
        let (y, cache) =
            self.attention
                .forward((self.attention_norm.forward(x), Some(cache), true));

        // Residual
        x += y;

        // Cross Attention
        let (y, enc_states) = self.cross_attention.cross_attention_forward(
            self.cross_attention_norm.forward(x),
            encoded,
            encoded,
        );

        // Residual
        x += y;

        // Feed Forward
        let y = self.ff.forward(self.feed_forward_norm.forward(x));

        // Residual
        (x + y, enc_states, cache)
    }
}

impl DecoderTransformerBlock {
    fn new(cx: &mut Graph) -> Self {
        Self {
            attention: SelfAttention::new(D_MODEL, cx),
            attention_norm: LayerNorm::new(D_MODEL, true, true, true, 1e-5, cx),
            cross_attention: SelfAttention::new(D_MODEL, cx),
            cross_attention_norm: LayerNorm::new(D_MODEL, true, true, true, 1e-5, cx),
            ff: (
                Linear::new_permuted(D_MODEL, DEC_FFN_DIM, true, cx),
                GeLU,
                Linear::new_permuted(DEC_FFN_DIM, D_MODEL, true, cx),
            ),
            feed_forward_norm: LayerNorm::new(D_MODEL, true, true, true, 1e-5, cx),
        }
    }
}

impl SerializeModule for DecoderTransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("self_attn_layer_norm", &self.attention_norm);
        s.module("encoder_attn", &self.cross_attention);
        s.module("encoder_attn_layer_norm", &self.cross_attention_norm);
        s.module("fc1", &self.ff.0);
        s.module("fc2", &self.ff.2);
        s.module("final_layer_norm", &self.feed_forward_norm);
    }
}

pub struct TextDecoder {
    // Embeddings
    pub embedding: Embedding,
    pub pos_embedding: GraphTensor,
    // Transformer layers
    pub layers: Vec<DecoderTransformerBlock>,
    // Final layer norm
    pub layer_norm: LayerNorm,
}

impl Module<(GraphTensor, GraphTensor, &[KVCache])> for TextDecoder {
    type Output = (
        GraphTensor,
        Vec<KVCache>, // Encoder projected states
        Vec<KVCache>, // Decoder KV cache
    );
    fn forward(
        &self,
        (enc_output, input, cache): (GraphTensor, GraphTensor, &[KVCache]),
    ) -> Self::Output {
        let (_, cur_dec_seq) = input.dims2();
        let (_, _, _, prev_dec_seq) = cache[0].0.dims4();

        // Embed text
        let mut x = self.embedding.forward(input);
        x += self
            .pos_embedding
            .slice((prev_dec_seq..cur_dec_seq + prev_dec_seq, ..))
            .contiguous()
            .expand_to(x.shape);

        // Run through layers and collect new caches
        let (mut new_caches, mut enc_states) = (vec![], vec![]);
        let (mut new_cache, mut enc_state);
        for (i, layer) in self.layers.iter().enumerate() {
            (x, enc_state, new_cache) = layer.forward((x, enc_output, cache[i]));
            new_caches.push(new_cache);
            enc_states.push(enc_state);
        }

        // Run through last norm and output projection
        let output = self.layer_norm.forward(x).matmul(
            self.embedding
                .weight
                .reshape((VOCAB_SIZE, D_MODEL))
                .permute((1, 0)),
        );

        (output, enc_states, new_caches)
    }
}

impl TextDecoder {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            embedding: Embedding::new_permuted(VOCAB_SIZE, D_MODEL, cx),
            pos_embedding: cx.tensor((MAX_TARGET_POSITION, D_MODEL)),
            layer_norm: LayerNorm::new(D_MODEL, true, true, true, 1e-5, cx),
            layers: (0..DEC_LAYERS)
                .map(|_| DecoderTransformerBlock::new(cx))
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
