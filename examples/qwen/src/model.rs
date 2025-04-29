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
pub const N_ATTENTION_GROUPS: usize = N_HEADS / N_KV_HEADS;

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

fn apply_rotary_embeddings_ggml(input: GraphTensor, prev_seq: Expression) -> GraphTensor {
    assert_eq!(input.shape.len(), 4);
    let (_, h, s, d) = input.dims4();

    // 1. Inverse frequencies 1 / θ^(2k/D)  (θ == ROPE_THETA)
    let inv_freq = ROPE_THETA.pow((input.graph().arange(d / 2) * 2) / d); // [half]

    // 2. Positions = arange(s) + prev_seq  ➜ [S]
    let pos = input.graph().arange(s) + prev_seq;

    // 3. Compute angles, then cos & sin
    let freqs = pos.expand(1, 1).matmul(inv_freq.expand(0, 1)); // [S,half]

    // Rotate hidden dimension
    let x0 = input.slice((.., .., .., ..h / 2));
    let x1 = input.slice((.., .., .., h / 2..));
    let rotated = (-x1).concat_along(x0, 3);

    input * freqs.cos().expand_to(input.shape) + rotated * freqs.sin().expand_to(input.shape)
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
                x.matmul(self.q_proj)
                    .reshape((batch, seq, N_HEADS, HEAD_DIM)), // .diff("../../../../Desktop/q.bin", 1e-2),
            )
            .permute((0, 2, 1, 3));

        let keys = self
            .k_norm
            .forward(
                x.matmul(self.k_proj)
                    .reshape((batch, seq, N_KV_HEADS, HEAD_DIM)), // .diff("../../../../Desktop/k.bin", 1e-2),
            )
            .permute((0, 2, 1, 3));

        let values = x
            .matmul(self.v_proj)
            .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));
        // queries.diff("../../../../Desktop/q_normed.bin", 1e-2);
        // keys.diff("../../../../Desktop/k_normed.bin", 1e-2);
        // values.diff("../../../../Desktop/v.bin", 1e-2);

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_ggml(queries.contiguous(), prev_seq);
        let keys = apply_rotary_embeddings_ggml(keys.contiguous(), prev_seq);
        // queries.diff("../../../../Desktop/q_rot.bin", 1e-2);
        // keys.diff("../../../../Desktop/k_rot.bin", 1e-2);

        // Add KV cache
        let keys = k_cache.concat_along(keys, 2);
        let values = v_cache.concat_along(values, 2);

        // Repeat the KV States for Grouped-Query Attention
        let repeated_keys = keys.expand(2, N_ATTENTION_GROUPS);
        let repeated_values = values.expand(2, N_ATTENTION_GROUPS);
        // repeated_keys
        //     .contiguous()
        //     .diff("../../../../Desktop/k_repeat.bin", 1e-1);
        // repeated_values
        //     .contiguous()
        //     .diff("../../../../Desktop/v_repeat.bin", 1e-1);

        // Calculate attention weights
        let mut attention_weights = queries
            .reshape((batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM)) // Split query heads into groups
            .matmul(repeated_keys.permute((0, 1, 2, 4, 3)))
            / (HEAD_DIM as f32).sqrt();

        let attention_mask = self.k_proj.graph().triu(seq, 1) * f32::MIN;
        attention_weights += attention_mask
            .pad(((0, 0), (prev_seq, 0)))
            .expand(0, batch)
            .expand(1, N_KV_HEADS)
            .expand(2, N_ATTENTION_GROUPS);

        // Calculate final outputs
        let output = attention_weights
            .softmax(4)
            // Apply distribution to values
            .matmul(repeated_values)
            // Merge heads
            .permute((0, 3, 1, 2, 4))
            .reshape((batch, seq, HEAD_DIM * N_HEADS));
        let output = output
            // Apply output projection
            .matmul(self.o_proj.permute((1, 0)));
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
    }
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj", (HIDDEN_DIM, HEAD_DIM * N_HEADS)),
            k_proj: cx.named_tensor("K Proj", (HIDDEN_DIM, HEAD_DIM * N_KV_HEADS)),
            v_proj: cx.named_tensor("V Proj", (HIDDEN_DIM, HEAD_DIM * N_KV_HEADS)),
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
    pub attention: SelfAttention,
    pub attention_norm: LayerNorm,
    pub feed_forward: Mlp,
    pub feed_forward_norm: LayerNorm,
}

impl Module<(GraphTensor, KVCache)> for TransformerBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (mut x, cache): (GraphTensor, KVCache)) -> Self::Output {
        // Attention
        let normed = self.attention_norm.forward(x);
        let (y, cache) = self.attention.forward((normed, cache));
        // y.diff("../../../../Desktop/attn.bin", 1e-1);
        // Residual
        x += y;
        // x.diff("../../../../Desktop/res.bin", 1e-1);
        // Feed Forward
        let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));
        // y.diff("../../../../Desktop/mlp.bin", 1e-1);

        // Residual
        (x + y, cache)
    }
}

impl TransformerBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            attention: SelfAttention::new(cx),
            attention_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
            feed_forward: Mlp::new(HIDDEN_DIM, MLP_DIM, cx),
            feed_forward_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
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

pub struct Qwen {
    // Token embeddings
    pub embedding: Embedding,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Norm + LM head
    pub head: (LayerNorm, Linear),
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
        (self.head.forward(x), new_caches)
    }
}

impl Qwen {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            embedding: Embedding::new(VOCAB_SIZE, HIDDEN_DIM, cx),
            head: (
                LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-6, cx),
                Linear::new_permuted(HIDDEN_DIM, VOCAB_SIZE, false, cx),
            ),
            layers: (0..NUM_LAYERS).map(|_| TransformerBlock::new(cx)).collect(),
        }
    }
}

impl SerializeModule for Qwen {
    fn serialize(&self, s: &mut Serializer) {
        s.module("token_embd", &self.embedding);
        s.module("output_norm", &self.head.0);
        s.module("output", &self.head.1);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
