use std::f32;

use luminal::prelude::{binary::F32Pow, *};
use luminal_nn::{Embedding, LayerNorm, Linear};

// Qwen3 4B Config
pub const VOCAB_SIZE: usize = 151936;
pub const HIDDEN_DIM: usize = 2560;
pub const NUM_LAYERS: usize = 36;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const MLP_DIM: usize = 9728;
pub const ROPE_THETA: f32 = 1_000_000.;
pub const HEAD_DIM: usize = 128;

pub type KVCache = (GraphTensor, GraphTensor);

pub struct Mlp {
    pub gate_proj: Linear, // hidden -> intermediate
    pub down_proj: Linear, // intermediate -> hidden
    pub up_proj: Linear,   // hidden -> intermediate
}

impl Module<GraphTensor> for Mlp {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        let gate = self.gate_proj.forward(input).silu();
        let up = self.up_proj.forward(input) * gate;
        self.down_proj.forward(up)
    }
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
        Self {
            gate_proj: Linear::new_permuted(hidden, intermediate, false, cx),
            down_proj: Linear::new_permuted(intermediate, hidden, false, cx),
            up_proj: Linear::new_permuted(hidden, intermediate, false, cx),
        }
    }
}

impl SerializeModule for Mlp {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ffn_gate", &self.gate_proj);
        s.module("ffn_up", &self.up_proj);
        s.module("ffn_down", &self.down_proj);
    }
}

fn apply_rotary_embeddings(input: GraphTensor, prev_seq: Expression) -> GraphTensor {
    assert_eq!(input.shape.len(), 4);
    let (b, h, s, d) = input.dims4();

    // Get freqs
    let inv_freq = ROPE_THETA
        .pow(input.graph().arange(d / 2) * 2 / d)
        .reciprocal(); // [d / 2]
    let pos = input.graph().arange(s) + prev_seq; // [s]
    let freqs = pos.expand_dim(1, 1).matmul(inv_freq.expand_dim(0, 1)); // [s, d / 2]
    let freqs = freqs
        .concat_along(freqs, freqs.shape.last_axis())
        .expand((b, h, s, d))
        .contiguous(); // [b, h, s, d]

    // Rotate input
    let rotated = (-input.slice((.., .., .., d / 2..)))
        .concat_along(input.slice((.., .., .., ..d / 2)), input.shape.last_axis());

    // Combine
    input * freqs.cos() + rotated * freqs.sin()
}

pub struct SelfAttention {
    pub q_proj: GraphTensor, // Hidden -> hidden
    pub k_proj: GraphTensor, // Proj dim -> hidden
    pub v_proj: GraphTensor, // Proj dim -> hidden
    pub o_proj: GraphTensor, // Hidden -> hidden
    pub q_norm: LayerNorm,
    pub k_norm: LayerNorm,
}

impl Module<(GraphTensor, KVCache)> for SelfAttention {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
        // x: batch, seq, hidden
        // cache: batch, kv_heads, prev_seq, head_dim
        let (batch, seq, _) = x.dims3();
        let (_, _, prev_seq, _) = k_cache.dims4();
        // Apply the Projections
        let queries = self
            .q_norm
            .forward(
                x.matmul(self.q_proj.permute((1, 0)))
                    .reshape((batch, seq, N_HEADS, HEAD_DIM))
                    .contiguous(),
            )
            .permute((0, 2, 1, 3));

        let keys = self
            .k_norm
            .forward(
                x.matmul(self.k_proj.permute((1, 0)))
                    .reshape((batch, seq, N_KV_HEADS, HEAD_DIM)),
            )
            .permute((0, 2, 1, 3));

        let values = x
            .matmul(self.v_proj.permute((1, 0)))
            .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings(queries, prev_seq);
        let keys = apply_rotary_embeddings(keys, prev_seq);

        // Add KV cache
        let keys = k_cache.concat_along(keys, 2);
        let values = v_cache.concat_along(values, 2);

        // Repeat the KV States for Grouped-Query Attention
        let repeated_keys = keys.expand_dim(2, N_HEADS / N_KV_HEADS);
        let repeated_values = values.expand_dim(2, N_HEADS / N_KV_HEADS);

        // Calculate attention weights
        let mut attention_weights = queries
            .reshape((batch, N_KV_HEADS, N_HEADS / N_KV_HEADS, seq, HEAD_DIM)) // Split query heads into groups
            .matmul(repeated_keys.permute((0, 1, 2, 4, 3)))
            / (HEAD_DIM as f32).sqrt();

        let attention_mask = self.k_proj.graph().triu(seq, 1) * f32::MIN;
        attention_weights += attention_mask
            .pad(((0, 0), (prev_seq, 0)))
            .expand_dim(0, batch)
            .expand_dim(1, N_KV_HEADS)
            .expand_dim(2, N_HEADS / N_KV_HEADS);

        // Calculate final outputs
        let output = attention_weights
            .softmax(4)
            // Apply distribution to values
            .matmul(repeated_values)
            // Merge heads
            .permute((0, 3, 1, 2, 4))
            .reshape((batch, seq, HEAD_DIM * N_HEADS));
        // Apply output projection
        let output = output.matmul(self.o_proj.permute((1, 0)));
        (output, (keys, values))
    }
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj", (HEAD_DIM * N_HEADS, HIDDEN_DIM)),
            k_proj: cx.named_tensor("K Proj", (HEAD_DIM * N_KV_HEADS, HIDDEN_DIM)),
            v_proj: cx.named_tensor("V Proj", (HEAD_DIM * N_KV_HEADS, HIDDEN_DIM)),
            o_proj: cx.named_tensor("O Proj", (HIDDEN_DIM, HEAD_DIM * N_HEADS)),
            q_norm: LayerNorm::new(HEAD_DIM, true, false, false, 1e-6, cx),
            k_norm: LayerNorm::new(HEAD_DIM, true, false, false, 1e-6, cx),
        }
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("attn_q/weight", self.q_proj);
        s.tensor("attn_v/weight", self.v_proj);
        s.tensor("attn_k/weight", self.k_proj);
        s.tensor("attn_output/weight", self.o_proj);
        s.module("attn_q_norm", &self.q_norm);
        s.module("attn_k_norm", &self.k_norm);
    }
}

pub struct TransformerBlock {
    pub attn: SelfAttention,
    pub attn_norm: LayerNorm,
    pub ff: Mlp,
    pub ff_norm: LayerNorm,
}

impl Module<(GraphTensor, KVCache)> for TransformerBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (mut x, cache): (GraphTensor, KVCache)) -> Self::Output {
        let normed = self.attn_norm.forward(x);
        let (y, cache) = self.attn.forward((normed, cache));
        x += y;
        let y = self.ff.forward(self.ff_norm.forward(x));
        (x + y, cache)
    }
}

impl TransformerBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            attn: SelfAttention::new(cx),
            attn_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
            ff: Mlp::new(HIDDEN_DIM, MLP_DIM, cx),
            ff_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
        }
    }
}

impl SerializeModule for TransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("", &self.attn);
        s.module("attn_norm", &self.attn_norm);
        s.module("ffn_norm", &self.ff_norm);
        s.module("", &self.ff);
    }
}

pub struct Qwen {
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
}

impl Module<(GraphTensor, &[KVCache])> for Qwen {
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(&self, (input, cache): (GraphTensor, &[KVCache])) -> Self::Output {
        // Embed tokens
        let mut x = self.embedding.forward(input);

        // Run through layers and collect new caches
        let mut new_caches = vec![];
        let mut new_cache;
        for (i, layer) in self.layers.iter().enumerate() {
            (x, new_cache) = layer.forward((x, cache[i]));
            new_caches.push(new_cache);
        }

        // Run through last norm and output projection
        (self.embedding.reverse(self.norm.forward(x)), new_caches)
    }
}

impl Qwen {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            embedding: Embedding::new(VOCAB_SIZE, HIDDEN_DIM, cx),
            norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
            layers: (0..NUM_LAYERS).map(|_| TransformerBlock::new(cx)).collect(),
        }
    }
}

impl SerializeModule for Qwen {
    fn serialize(&self, s: &mut Serializer) {
        s.module("token_embd", &self.embedding);
        s.module("output_norm", &self.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
