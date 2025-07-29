use luminal::prelude::*;
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

fn apply_rotary_embeddings_cuda(input: GraphTensor, prev_seq: Expression) -> GraphTensor {
    let (batch, n_heads, seq, head_dim) = input.dims4();
    let (n_heads_kernel, seq_kernel, head_dim_kernel, prev_seq_kernel) = (
        n_heads.to_kernel(),
        seq.to_kernel(),
        head_dim.to_kernel(),
        prev_seq.to_kernel(),
    );
    let [x] = custom_kernel(
        &[input],
        Kernel {
            code: format!(
                "
#include <math_constants.h>

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ x,  // [B,H,S,D]
    float* __restrict__ y,        // [B,H,S,D] (can be same as x)
    const size_t const_p, const size_t const_s
) {{
    // Each block handles one (b,h,s) row; threads iterate over i in [0, D/2)
    int row = blockIdx.x;
    int s = row % {seq_kernel};
    int tmp = row / {seq_kernel};
    int h = tmp % {n_heads_kernel};
    int b = tmp / {n_heads_kernel};

    const float rope_base = 500000.0f;

    const int pairs = {head_dim_kernel} >> 1; // D/2
    const size_t row_base = (((size_t)b * {n_heads_kernel} + h) * {seq_kernel} + s) * {head_dim_kernel};

    const float pos = (float)({prev_seq_kernel} + s);

    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {{
        const size_t even_idx = row_base + (size_t)(2 * i);
        const size_t odd_idx  = even_idx + 1;

        // Load pair
        float x0 = x[even_idx];
        float x1 = x[odd_idx];

        // Angle = pos * inv_freq[i]
        float inv = 1.0f / powf(rope_base, (2.0f*i) / (float){head_dim_kernel});
        float angle = pos * inv;

        float sn, cs;
        __sincosf(angle, &sn, &cs);

        // Rotate
        float out0 = x0 * cs - x1 * sn; // even
        float out1 = x0 * sn + x1 * cs; // odd

        // Store (supports in-place)
        y[even_idx] = out0;
        y[odd_idx]  = out1;
    }}
}}"
            ),
            grid: ((seq * n_heads).into(), 1.into(), 1.into()),
            threadblock: (64.into(), 1.into(), 1.into()),
            smem: 0.into(),
            outputs: vec![input.shape.n_elements()],
        },
        [(batch, n_heads, seq, head_dim)],
        input.graph(),
    );
    x
}

pub struct SelfAttention {
    pub q_proj: GraphTensor, // Hidden -> hidden
    pub k_proj: GraphTensor, // Proj dim -> hidden
    pub v_proj: GraphTensor, // Proj dim -> hidden
    pub o_proj: GraphTensor, // Hidden -> hidden
}

fn attention_qkv_cuda(
    queries: GraphTensor, // [B,Hk,G,S,D]
    keys: GraphTensor,    // [B,Hk,prev_seq + S,D]
    values: GraphTensor,  // [B,Hk,prev_seq + S,D]
    prev_seq: Expression,
) -> GraphTensor {
    let (batch, hk, groups, seq, head_dim) = queries.dims5();
    let (_bk, _hk_k, _t_k, _d_k) = keys.dims4();
    let (_bv, _hk_v, _t_v, _d_v) = values.dims4();

    let (
        batch_kernel,
        n_kv_heads_kernel,
        n_groups_kernel,
        seq_kernel,
        head_dim_kernel,
        prev_seq_kernel,
    ) = (
        batch.to_kernel(),
        hk.to_kernel(),
        groups.to_kernel(),
        seq.to_kernel(),
        head_dim.to_kernel(),
        prev_seq.to_kernel(),
    );

    let tb_x: usize = 64;
    let hidden_dim = hk * groups * head_dim;

    let [out] = custom_kernel(
        &[queries, keys, values],
        Kernel {
            code: format!(
                "
#include <math_constants.h>
#include <math_functions.h>

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ q,   // [B,Hk,G,S,D]
    const float* __restrict__ k,   // [B,Hk,T,D], T = prev_seq + S
    const float* __restrict__ v,   // [B,Hk,T,D]
    float* __restrict__ y,         // [B,S,(Hk*G)*D]
    const size_t const_p, const size_t const_s
) {{
    const int B   = {batch_kernel};
    const int Hk  = {n_kv_heads_kernel};
    const int G   = {n_groups_kernel};
    const int S   = {seq_kernel};
    const int D   = {head_dim_kernel};
    const int T   = {prev_seq_kernel} + S;
    const int Hq  = Hk * G;
    const float scale = rsqrtf((float)D);

    // one block per (b, hk, g, s)
    int row = blockIdx.x;
    int s   = row % S;
    int tmp = row / S;
    int g   = tmp % G;  tmp /= G;
    int hk  = tmp % Hk; tmp /= Hk;
    int b   = tmp;
    if (b >= B) return;

    // dynamic shared memory for reductions/broadcasts
    extern __shared__ float red[];

    // bases
    size_t q_row_base = (((((size_t)b * Hk + (size_t)hk) * G + (size_t)g) * S + (size_t)s) * (size_t)D);
    size_t out_row_base = (((size_t)b * (size_t)S + (size_t)s) * (size_t)(Hq * D))
                         + (((size_t)hk * (size_t)G + (size_t)g) * (size_t)D);

    // zero output slice
    for (int d = threadIdx.x; d < D; d += blockDim.x) {{
        y[out_row_base + d] = 0.0f;
    }}
    __syncthreads();

    // Pass 1: max score across t
    float max_score = -CUDART_INF_F;
    for (int t = 0; t < T; ++t) {{
        size_t k_row_base = ((((size_t)b * Hk + (size_t)hk) * (size_t)T + (size_t)t) * (size_t)D);
        float partial = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {{
            partial += q[q_row_base + d] * k[k_row_base + d];
        }}
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {{
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }}
        if (threadIdx.x == 0) {{
            float score = red[0] * scale;
            if (t > ({prev_seq_kernel} + s)) score = -CUDART_INF_F; // causal with left pad
            if (score > max_score) max_score = score;
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) red[0] = max_score;
    __syncthreads();
    float max_s = red[0];

    // Pass 2: denom = sum_t exp(score - max_s)  (use separate accumulator)
    float denom_acc = 0.0f;
    for (int t = 0; t < T; ++t) {{
        size_t k_row_base = ((((size_t)b * Hk + (size_t)hk) * (size_t)T + (size_t)t) * (size_t)D);
        float partial = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {{
            partial += q[q_row_base + d] * k[k_row_base + d];
        }}
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {{
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }}
        if (threadIdx.x == 0) {{
            float score = red[0] * scale;
            if (t > ({prev_seq_kernel} + s)) score = -CUDART_INF_F;
            denom_acc += expf(score - max_s);
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) red[0] = denom_acc;
    __syncthreads();
    float denom = red[0];

    // Pass 3: y += sum_t softmax(score)_t * v_t
    for (int t = 0; t < T; ++t) {{
        size_t kv_row_base = ((((size_t)b * Hk + (size_t)hk) * (size_t)T + (size_t)t) * (size_t)D);

        float partial = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {{
            partial += q[q_row_base + d] * k[kv_row_base + d];
        }}
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {{
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }}
        if (threadIdx.x == 0) {{
            float score = red[0] * scale;
            if (t > ({prev_seq_kernel} + s)) score = -CUDART_INF_F;
            red[0] = expf(score - max_s) / denom; // weight
        }}
        __syncthreads();
        float w = red[0];

        for (int d = threadIdx.x; d < D; d += blockDim.x) {{
            y[out_row_base + d] += w * v[kv_row_base + d];
        }}
        __syncthreads();
    }}
}}"
            ),
            grid: ((batch * hk * groups * seq).into(), 1.into(), 1.into()),
            threadblock: (tb_x.into(), 1.into(), 1.into()),
            smem: (tb_x * 4).into(),
            outputs: vec![batch * seq * hidden_dim],
        },
        [(batch, seq, hidden_dim)],
        queries.graph(),
    );
    out
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
        let queries = apply_rotary_embeddings_cuda(queries.contiguous(), prev_seq);
        let keys = apply_rotary_embeddings_cuda(keys.contiguous(), prev_seq);

        // Add KV cache
        let keys = k_cache.concat_along(keys, 2);
        let values = v_cache.concat_along(values, 2);

        let output = attention_qkv_cuda(
            queries.reshape((batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM)),
            keys,
            values,
            prev_seq,
        );

        // Apply output projection
        (bmm_col_major_b(output, self.o_proj), (keys, values))
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
                grid: (sequence_length, (HIDDEN_DIM / 256).into(), 1.into()),
                threadblock: (1.into(), 256.into(), 1.into()),
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
        x = bmm_col_major_b(self.head_norm.forward(x), self.head_proj.weight);
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
