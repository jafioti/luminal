use luminal::prelude::{binary::F32Pow, *};
use luminal_2::{custom_kernel, Kernel};
use luminal_nn::{LayerNorm, Linear};

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
        let up = bmm_col_major_b(input, self.up_proj.weight)
            * bmm_col_major_b(input, self.gate_proj.weight).swish();
        bmm_col_major_b(up, self.down_proj.weight)
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
    assert_eq!(input.shape.len(), 4); // batch, n_heads, seq, head_dim
    let (batch, n_heads, seq, head_dim) = input.dims4();
    // Get freqs
    let freqs = (input.graph().arange(head_dim / 2) * 2.0) / (head_dim.to_usize().unwrap() as f32);
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let pos = input.graph().arange(seq) + prev_seq;
    let emb = pos.expand_dim(1, 1).matmul(inv_freqs.expand_dim(0, 1));

    // Split input into evens and odds
    let split = input.reshape((batch, n_heads, seq, head_dim / 2, 2));
    let x0 = split.slice((.., .., .., .., ..1));
    let x1 = split.slice((.., .., .., .., 1..));

    // Apply sin and cos embeddings
    let x0_out = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
    let x1_out = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

    // Combine back into output
    x0_out.concat_along(x1_out, 4).reshape(input.shape)
}

pub struct SelfAttention {
    pub q_proj: GraphTensor, // Hidden -> hidden
    pub k_proj: GraphTensor, // Proj dim -> hidden
    pub v_proj: GraphTensor, // Proj dim -> hidden
    pub o_proj: GraphTensor, // Hidden -> hidden
}

impl Module<(GraphTensor, KVCache)> for SelfAttention {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
        let (batch, seq, _hidden) = x.dims3();
        let (_, _kv_heads, prev_seq, _head_dim) = k_cache.dims4();

        // Apply the Projections
        let queries = bmm_col_major_b(x, self.q_proj)
            .reshape((batch, seq, N_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        let keys = bmm_col_major_b(x, self.k_proj)
            .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        let values = bmm_col_major_b(x, self.v_proj)
            .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_ggml(queries, prev_seq);
        let keys = apply_rotary_embeddings_ggml(keys, prev_seq);

        // Add KV cache
        let keys = k_cache.concat_along(keys, 2);
        let values = v_cache.concat_along(values, 2);

        // Repeat the KV States for Grouped-Query Attention
        let repeated_keys = keys.expand_dim(2, N_ATTENTION_GROUPS);
        let repeated_values = values.expand_dim(2, N_ATTENTION_GROUPS);

        let mut attention_weights = queries
            .reshape((batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM))
            .matmul(repeated_keys.permute((0, 1, 2, 4, 3)))
            / (HEAD_DIM as f32).sqrt();

        // Causal mask
        let attention_mask = self.k_proj.graph().triu(seq, 1) * f16::MIN.to_f32();
        attention_weights += attention_mask
            .pad_along(prev_seq, 0, 1)
            .expand_dim(0, batch)
            .expand_dim(1, N_KV_HEADS)
            .expand_dim(2, N_ATTENTION_GROUPS);

        // Calculate final outputs
        let output = attention_weights
            .softmax(4)
            // Apply distribution to values
            .matmul(repeated_values)
            // Merge heads
            .permute((0, 3, 1, 2, 4))
            .reshape((batch, seq, HIDDEN_DIM));
        // Apply output projection
        let output = bmm_col_major_b(output, self.o_proj);
        (output, (keys, values))
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
    // pub embedding: Embedding,
    pub embedding: GraphTensor,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Norm + LM head
    pub head_norm: LayerNorm,
    pub head_proj: Linear,
}

impl Module<(GraphTensor, &[KVCache])> for Llama {
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(&self, (input, cache): (GraphTensor, &[KVCache])) -> Self::Output {
        // Embed tokens
        let (batch, sequence_length) = input.dims2();
        let [mut x] = custom_kernel(
            &[input, self.embedding],
            Kernel {
                code: format!(
                    "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel_name(const float *inp, const float *weights, float *out) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < {VOCAB_SIZE} && y < {HIDDEN_DIM}) {{
        out[x * {HIDDEN_DIM} + y] = weights[(int)inp[x] * {HIDDEN_DIM} + y];
    }}
}}"
                ),
                grid: (
                    sequence_length,
                    Expression::from(HIDDEN_DIM),
                    Expression::from(1),
                ),
                threadblock: (
                    Expression::from(1),
                    Expression::from(1),
                    Expression::from(1),
                ),
                smem: Expression::from(0),
                outputs: vec![sequence_length * HIDDEN_DIM],
            },
            [(batch, sequence_length, HIDDEN_DIM)],
            input.graph(),
        );

        // Run through layers and collect new caches
        let mut new_caches = vec![];
        let mut new_cache;
        for (layer, cache) in self.layers.iter().zip(cache) {
            (x, new_cache) = layer.forward((x, *cache));
            new_caches.push(new_cache);
        }

        // Run through last norm and output projection
        x = self.head_norm.forward(x);
        [x] = custom_kernel(
            &[x, self.head_proj.weight],
            Kernel {
                code: format!(
                    "
extern \"C\" __global__ void kernel_name(
   	const float* A,
    const float* B,
    float* C,
    const size_t const_p,
    const size_t const_s
)
{{
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    const float* a = A + m * {HIDDEN_DIM};
    const float* b = B + n * {HIDDEN_DIM};
    float acc = 0.f;
    for (int k = 0; k < {HIDDEN_DIM}; ++k) {{
       	acc += a[k] * b[k];
    }}

    C[m * {VOCAB_SIZE} + n] = acc;
}}
"
                ),
                grid: ((VOCAB_SIZE / 64).into(), sequence_length, 1.into()),
                threadblock: (64.into(), 1.into(), 1.into()),
                smem: 0.into(),
                outputs: vec![sequence_length * VOCAB_SIZE],
            },
            [(batch, sequence_length, VOCAB_SIZE)],
            input.graph(),
        );
        (x, new_caches)
    }
}

impl Llama {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            // embedding: Embedding::new(VOCAB_SIZE, HIDDEN_DIM, cx),
            embedding: cx.tensor((VOCAB_SIZE, HIDDEN_DIM)),
            head_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
            head_proj: Linear::new_permuted(HIDDEN_DIM, VOCAB_SIZE, false, cx),
            layers: (0..NUM_LAYERS).map(|_| TransformerBlock::new(cx)).collect(),
        }
    }
}

impl SerializeModule for Llama {
    fn serialize(&self, s: &mut Serializer) {
        // s.module("token_embd", &self.embedding);
        s.tensor("token_embd/weight", self.embedding);
        s.module("output_norm", &self.head_norm);
        s.module("output", &self.head_proj);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("blk/{i}"), layer);
        }
    }
}

fn bmm_col_major_b(a: GraphTensor, b: GraphTensor) -> GraphTensor {
    let (batch, m, k) = a.dims3();
    let (n, _) = b.dims2();
    let (m, n, k) = (m.to_kernel(), n.to_kernel(), k.to_kernel());
    let [out] = custom_kernel(
        &[a, b],
        Kernel {
            code: format!(
                "
// C[m,n] = sum_k A[m,k] * B[n,k]
// A: [M,K] row-major, B: [N,K] row-major, C: [M,N] row-major
__inline__ __device__ float warp_sum(float v, unsigned mask){{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(mask, v, o);
    return v;
}}

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const size_t const_p,
    const size_t const_s)
{{
    const int n = blockIdx.x;                 // one output column per block
    if (n >= {n}) return;

    const int TM = 8;                         // rows per tile (tune: 8 or 16)
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned fullmask = __activemask();

    extern __shared__ float s[];              // size = TM * nwarps floats
    float* warpbuf = s;

    for (int m0 = 0; m0 < {m}; m0 += TM) {{
        const int Me = min((int)TM, (int)({m} - m0));

        // Register accumulators for a tile of rows
        float acc[8];                         // TM=8
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;

        // Stride over K
        for (int k = tid; k < {k}; k += blockDim.x) {{
            const float x = B[n * {k} + k];     // B[n,k]
            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
                if (i < Me)
                    acc[i] += A[(m0 + i) * {k} + k] * x;
            }}
        }}

        // Reduce within each warp, per row in the tile
        #pragma unroll
        for (int i = 0; i < 8; ++i) {{
            float v = (i < Me) ? acc[i] : 0.f;
            v = warp_sum(v, fullmask);
            if (lane == 0 && i < Me) warpbuf[i * nwarps + wid] = v;
        }}
        __syncthreads();

        // First warp reduces across warps and writes results
        if (wid == 0) {{
            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
                if (i >= Me) break;
                float v = (lane < nwarps) ? warpbuf[i * nwarps + lane] : 0.f;
                v = warp_sum(v, 0xffffffff);
                if (lane == 0) C[(m0 + i) * {n} + n] = v;
            }}
        }}
        __syncthreads();
    }}
}}
   "
            ),
            grid: (b.dims()[0], 1.into(), 1.into()),
            threadblock: (256.into(), 1.into(), 1.into()),
            smem: (8 * ((256 + 31) / 32) * size_of::<f32>()).into(),
            outputs: vec![batch * a.dims()[1] * b.dims()[0]],
        },
        [(batch, a.dims()[1], b.dims()[0])],
        a.graph(),
    );
    out
}
