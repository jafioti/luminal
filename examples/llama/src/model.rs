use luminal::prelude::{binary::F32Pow, *};
use luminal_2::{custom_kernel, Kernel};
use luminal_nn::{Embedding, LayerNorm, Linear};

// Llama3 8B Config
pub const VOCAB_SIZE: usize = 128256;
pub const HIDDEN_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const MLP_DIM: usize = 14336;

pub const N_ATTENTION_GROUPS: usize = N_HEADS / N_KV_HEADS;
pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS;
pub const ATTN_PROJ_DIM: usize = HEAD_DIM * N_KV_HEADS;

pub type KVCache = (GraphTensor, GraphTensor);

pub struct Mlp {
    pub gate_proj: Linear, // hidden -> intermediate
    pub down_proj: Linear, // intermediate -> hidden
    pub up_proj: Linear,   // hidden -> intermediate
}

impl Module<GraphTensor> for Mlp {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        let gate = self.gate_proj.forward(input).swish();
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
    assert_eq!(input.shape.len(), 3); // batch, n_heads, seq, head_dim
    let (n_heads, seq, head_dim) = input.dims3();
    // Get freqs
    let freqs = (input.graph().arange(head_dim / 2) * 2.0) / (head_dim.to_usize().unwrap() as f32);
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let pos = input.graph().arange(seq) + prev_seq;
    let emb = pos.expand_dim(1, 1).matmul(inv_freqs.expand_dim(0, 1));

    // Split input into evens and odds
    let split = input.reshape((n_heads, seq, head_dim / 2, 2));
    let x0 = split.slice((.., .., .., ..1));
    let x1 = split.slice((.., .., .., 1..));

    // Apply sin and cos embeddings
    let x0_out = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
    let x1_out = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

    // Combine back into output
    x0_out.concat_along(x1_out, 3).reshape(input.shape)
}

pub struct SelfAttention {
    pub q_proj: GraphTensor, // Hidden -> hidden
    pub k_proj: GraphTensor, // Proj dim -> hidden
    pub v_proj: GraphTensor, // Proj dim -> hidden
    pub o_proj: GraphTensor, // Hidden -> hidden
}

// impl Module<(GraphTensor)> for SelfAttention {
//     type Output = (GraphTensor);
//     fn forward(&self, (x): (GraphTensor)) -> Self::Output {
impl Module<(GraphTensor, KVCache)> for SelfAttention {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
        // x: batch, seq, hidden
        // cache: batch, kv_heads, prev_seq, head_dim
        let (seq, _) = x.dims2();
        let (_, prev_seq, _) = k_cache.dims3();
        // Apply the Projections
        let queries = x
            .matmul(self.q_proj.permute((1, 0)))
            .reshape((seq, N_HEADS, HEAD_DIM))
            .permute((1, 0, 2));

        let keys = x
            .matmul(self.k_proj.permute((1, 0)))
            .reshape((seq, N_KV_HEADS, HEAD_DIM))
            .permute((1, 0, 2));

        let values = x
            .matmul(self.v_proj.permute((1, 0)))
            .reshape((seq, N_KV_HEADS, HEAD_DIM))
            .permute((1, 0, 2));

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_ggml(queries, prev_seq);
        let keys = apply_rotary_embeddings_ggml(keys, prev_seq);

        // Add KV cache
        let keys = k_cache.concat_along(keys, 1);
        let values = v_cache.concat_along(values, 1);

        // Repeat the KV States for Grouped-Query Attention
        let repeated_keys = keys.expand_dim(1, N_ATTENTION_GROUPS);
        let repeated_values = values.expand_dim(1, N_ATTENTION_GROUPS);

        // Calculate attention weights
        let mut attention_weights = queries
            .reshape((N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM)) // Split query heads into groups
            .matmul(repeated_keys.permute((0, 1, 3, 2)))
            / (HEAD_DIM as f32).sqrt();

        let attention_mask = self.k_proj.graph().triu(seq, 1) * f16::MIN.to_f32();
        attention_weights += attention_mask
            .pad((prev_seq, 0))
            .expand_dim(0, N_KV_HEADS)
            .expand_dim(1, N_ATTENTION_GROUPS);

        // Calculate final outputs
        let output = attention_weights
            .softmax(3)
            // Apply distribution to values
            .matmul(repeated_values)
            // Merge heads
            .permute((2, 0, 1, 3))
            .reshape((seq, HIDDEN_DIM))
            // Apply output projection
            .matmul(self.o_proj.permute((1, 0)));
        (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
    }
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj", (HIDDEN_DIM, HIDDEN_DIM)),
            k_proj: cx.named_tensor("K Proj", (ATTN_PROJ_DIM, HIDDEN_DIM)),
            v_proj: cx.named_tensor("V Proj", (ATTN_PROJ_DIM, HIDDEN_DIM)),
            o_proj: cx.named_tensor("O Proj", (HIDDEN_DIM, HIDDEN_DIM)),
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
    pub attention_norm: LayerNorm,
    pub feed_forward: Mlp,
    pub feed_forward_norm: LayerNorm,
}
// impl Module<(GraphTensor)> for TransformerBlock {
//     type Output = (GraphTensor);
//     fn forward(&self, (mut x): (GraphTensor)) -> Self::Output {
impl Module<(GraphTensor, KVCache)> for TransformerBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (mut x, cache): (GraphTensor, KVCache)) -> Self::Output {
        // Attention
        let (y, cache) = self
            .attention
            .forward((self.attention_norm.forward(x), cache));

        // Residual
        x += y;

        // Feed Forward
        let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

        // Residual
        (x + y, cache)
    }
}

impl TransformerBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            attention: SelfAttention::new(cx),
            attention_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
            feed_forward: Mlp::new(HIDDEN_DIM, MLP_DIM, cx),
            feed_forward_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
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

pub struct Llama {
    // Token embeddings
    pub embedding: GraphTensor,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Norm + LM head
    pub head: (LayerNorm, Linear),
}

impl Module<(GraphTensor, &[KVCache])> for Llama {
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(&self, (input, cache): (GraphTensor, &[KVCache])) -> Self::Output {
        // Embed tokens
        let sequence_length = input.dims1();
        let [mut x] = custom_kernel(
            &[input, self.embedding],
            Kernel {
                code: "#include <metal_stdlib>
using namespace metal;
kernel void kernel(device float *inp [[buffer(0)]], device float *weights [[buffer(1)]], device float *out [[buffer(2)]], device int& n_embeddings [[buffer(3)]], device int& embedding_dim [[buffer(4)]], uint2 i_ [[thread_position_in_grid]]) {
    if (i_.x < n_embeddings && i_.y < embedding_dim) {
        out[i_.x * embedding_dim + i_.y] = weights[(int)inp[i_.x] * embedding_dim + i_.y];
    }
}".to_string(),
                grid: (
                    sequence_length,
                    Expression::from(HIDDEN_DIM),
                    Expression::from(1),
                ),
                threadblock: (
                    Expression::from(16),
                    Expression::from(16),
                    Expression::from(1),
                ),
                smem: Expression::from(0),
                outputs: vec![sequence_length * HIDDEN_DIM],
            },
            [(sequence_length, HIDDEN_DIM)],
            input.graph(),
        );

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

impl Llama {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            embedding: cx.tensor((VOCAB_SIZE, HIDDEN_DIM)),
            head: (
                LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
                Linear::new_permuted(HIDDEN_DIM, VOCAB_SIZE, false, cx),
            ),
            layers: (0..NUM_LAYERS).map(|_| TransformerBlock::new(cx)).collect(),
        }
    }
}

impl SerializeModule for Llama {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("token_embd/weight", self.embedding);
        s.module("output_norm", &self.head.0);
        s.module("output", &self.head.1);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}
