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
        let a = proj_swiglu_cuda(input, self.up_proj.weight, self.gate_proj.weight);
        bmm_col_major_b(a, self.down_proj.weight)
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
        let queries = bmm_col_major_b(x, self.q_proj).reshape((batch, seq, N_HEADS, HEAD_DIM));
        let keys = bmm_col_major_b(x, self.k_proj).reshape((batch, seq, N_KV_HEADS, HEAD_DIM));
        let values = bmm_col_major_b(x, self.v_proj)
            .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
            .permute((0, 2, 1, 3));

        // Rotary embed queries and keys
        let queries = apply_rotary_embeddings_cuda(queries, prev_seq);
        let keys = apply_rotary_embeddings_cuda(keys, prev_seq);

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
        let (y, cache) = self.attention.forward((
            layernorm_cuda(x, self.attention_norm.weight.unwrap()),
            cache,
        ));

        // Residual
        x += y;

        // Feed Forward
        let y = self
            .feed_forward
            .forward(layernorm_cuda(x, self.feed_forward_norm.weight.unwrap()));

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
        let s = sequence_length.to_kernel();
        let mut x = custom_kernel(
            &[input, self.embedding],
            Kernel {
                code: format!(
                    "
        #include <math_functions.h>

        extern \"C\" __global__ void kernel_name(
            const float* __restrict__ inp,      // [B*S] token ids as float (cast to int)
            const float* __restrict__ weights,  // [VOCAB_SIZE, HIDDEN_DIM] row-major
            float* __restrict__ out,             // [B,S,HIDDEN_DIM] row-major
            const size_t const_p, const size_t const_s
        ){{
            const int H = {HIDDEN_DIM};
            const int total = {batch} * {s};   // B*S
            int pos = blockIdx.x;                            // one block per token position
            if (pos >= total) return;

            // token id
            int tok = (int)inp[pos];                         // assumes valid 0..VOCAB_SIZE-1

            // row pointers
            const float* __restrict__ row = weights + (size_t)tok * H;
            float* __restrict__ dst       = out     + (size_t)pos * H;

            // vectorized copy: float4 chunks then scalar tail
            int h4 = H >> 2;                                 // number of float4s
            const float4* __restrict__ r4 = reinterpret_cast<const float4*>(row);
            float4* __restrict__ d4       = reinterpret_cast<float4*>(dst);

            // threads stride across float4 chunks
            for (int i = threadIdx.x; i < h4; i += blockDim.x) {{
                float4 v = r4[i];
                d4[i] = v;
            }}

            // tail (H % 4)
            for (int h = (h4 << 2) + threadIdx.x; h < H; h += blockDim.x) {{
                dst[h] = row[h];
            }}
        }}"
                ),
                // one block per token position
                grid: ((batch * sequence_length).into(), 1.into(), 1.into()),
                // threads stride along hidden; 256 is a good default
                threadblock: (256.into(), 1.into(), 1.into()),
                smem: 0.into(),
                outputs: vec![batch * sequence_length * HIDDEN_DIM],
            },
            // output shape [B,S,H]
            (batch, sequence_length, HIDDEN_DIM),
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
        x = bmm_col_major_b(
            layernorm_cuda(x, self.head_norm.weight.unwrap()),
            self.head_proj.weight,
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
    let (m_k, n_k, k_k) = (m.to_kernel(), n.to_kernel(), k.to_kernel());

    let out = custom_kernel(
        &[a, b],
        Kernel {
            code: format!(
                "
// C[m,n] = sum_k A[m,k] * B[n,k]
// A: [M,K] row-major, B: [N,K] row-major, C: [M,N] row-major
#include <math_functions.h>

__inline__ __device__ float warp_sum(float v, unsigned mask){{
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}}

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ A,   // [M,K]
    const float* __restrict__ B,   // [N,K]
    float* __restrict__ C,         // [M,N]
    const size_t const_p, const size_t const_s)
{{
    const int N = {n_k};
    const int M = {m_k};
    const int K = {k_k};

    const int ncol = blockIdx.x;         // one output column per block
    if (ncol >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned fullmask = __activemask();

    // Fast path: M == 1 (decoding). No shared memory needed.
    if (M == 1) {{
        const float* __restrict__ a0 = A;                // row 0
        const float* __restrict__ bn = B + (size_t)ncol * K;

        float acc = 0.f;

        // Vectorized over K (float4) if size multiple of 4
        const int K4 = (K >> 2) << 2;
        const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a0);
        const float4* __restrict__ b4 = reinterpret_cast<const float4*>(bn);

        for (int k4 = tid; k4 < (K4 >> 2); k4 += blockDim.x) {{
            float4 av = a4[k4];
            float4 bv = b4[k4];
            acc = fmaf(av.x, bv.x, acc);
            acc = fmaf(av.y, bv.y, acc);
            acc = fmaf(av.z, bv.z, acc);
            acc = fmaf(av.w, bv.w, acc);
        }}
        for (int k = K4 + tid; k < K; k += blockDim.x) {{
            acc = fmaf(a0[k], bn[k], acc);
        }}

        acc = warp_sum(acc, fullmask);
        __shared__ float warpbuf[8];   // up to 8 warps (blockDim.x<=256)
        if (lane == 0) warpbuf[wid] = acc;
        __syncthreads();

        if (wid == 0) {{
            float v = (lane < nwarps) ? warpbuf[lane] : 0.f;
            v = warp_sum(v, 0xffffffff);
            if (lane == 0) {{
                float outv = v;
                C[ncol] = outv;        // C[0,n]
            }}
        }}
        return;
    }}

    // General path: M >= 1 (prompt processing). Tile rows by TM and reduce across warps.
    const int TM = 8; // rows per tile
    extern __shared__ float s[]; // size = TM * nwarps floats
    float* warpbuf = s;

    for (int m0 = 0; m0 < M; m0 += TM) {{
        const int Me = min(TM, M - m0);

        float acc[TM];
        #pragma unroll
        for (int i = 0; i < TM; ++i) acc[i] = 0.f;

        const float* __restrict__ bn = B + (size_t)ncol * K;
        const int K4 = (K >> 2) << 2;
        const float4* __restrict__ b4 = reinterpret_cast<const float4*>(bn);

        for (int k4 = tid; k4 < (K4 >> 2); k4 += blockDim.x) {{
            float4 bv = b4[k4];
            int k = k4 << 2;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {{
                if (i >= Me) break;
                const float* __restrict__ ai = A + (size_t)(m0 + i) * K + k;
                float4 av = *reinterpret_cast<const float4*>(ai);
                acc[i] = fmaf(av.x, bv.x, acc[i]);
                acc[i] = fmaf(av.y, bv.y, acc[i]);
                acc[i] = fmaf(av.z, bv.z, acc[i]);
                acc[i] = fmaf(av.w, bv.w, acc[i]);
            }}
        }}
        for (int k = K4 + tid; k < K; k += blockDim.x) {{
            float bx = bn[k];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {{
                if (i < Me) acc[i] = fmaf(A[(m0 + i) * K + k], bx, acc[i]);
            }}
        }}

        #pragma unroll
        for (int i = 0; i < TM; ++i) {{
            float v = (i < Me) ? acc[i] : 0.f;
            v = warp_sum(v, fullmask);
            if (lane == 0 && i < Me) warpbuf[i * nwarps + wid] = v;
        }}
        __syncthreads();

        if (wid == 0) {{
            #pragma unroll
            for (int i = 0; i < TM; ++i) {{
                if (i >= Me) break;
                float v = (lane < nwarps) ? warpbuf[i * nwarps + lane] : 0.f;
                v = warp_sum(v, 0xffffffff);
                if (lane == 0) {{
                    float outv = v;
                    C[(m0 + i) * N + ncol] = outv;
                }}
            }}
        }}
        __syncthreads();
    }}
}}"
            ),
            // one block per column n
            grid: (b.dims()[0], 1.into(), 1.into()),
            // keep 256 threads (8 warps)
            threadblock: (256.into(), 1.into(), 1.into()),
            // shared mem for general path: TM * nwarps * sizeof(float)
            smem: (8 * ((256 + 31) / 32) * size_of::<f32>()).into(),
            outputs: vec![batch * a.dims()[1] * b.dims()[0]],
        },
        (batch, a.dims()[1], b.dims()[0]),
        a.graph(),
    );
    out
}

fn layernorm_cuda(x: GraphTensor, weight: GraphTensor) -> GraphTensor {
    let (batch, seq, hidden) = x.dims3();
    let (b_k, s_k, h_k) = (batch.to_kernel(), seq.to_kernel(), hidden.to_kernel());
    let tb_x: usize = 256;

    let y = custom_kernel(
        &[x, weight],
        Kernel {
            code: format!(
                "
#include <math_functions.h>

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ x,       // [B,S,H]
    const float* __restrict__ gamma,   // [H]
    float* __restrict__ y,             // [B,S,H]
    const size_t const_p, const size_t const_s
){{
    const int B = {b_k};
    const int S = {s_k};
    const int H = {h_k};
    const float EPS = 1e-5f;

    int row = blockIdx.x;                  // one block per (b,s)
    if (row >= B * S) return;

    size_t base = (size_t)row * (size_t)H;

    // dynamic shared mem for reduction
    extern __shared__ float red[];

    // sum of squares over H
    float sumsq = 0.f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {{
        float v = x[base + h];
        sumsq += v * v;
    }}
    red[threadIdx.x] = sumsq;
    __syncthreads();

    // reduce to red[0]
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {{
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }}

    // compute inv_rms and broadcast via red[0]
    if (threadIdx.x == 0) {{
        float mean_sumsq = red[0] / (float)H;
        red[0] = rsqrtf(mean_sumsq + EPS);   // inv_rms
    }}
    __syncthreads();
    float inv_rms = red[0];

    // y = x * inv_rms * gamma
    for (int h = threadIdx.x; h < H; h += blockDim.x) {{
        y[base + h] = x[base + h] * inv_rms * gamma[h];
    }}
}}"
            ),
            grid: ((batch * seq).into(), 1.into(), 1.into()),
            threadblock: (tb_x.into(), 1.into(), 1.into()),
            smem: (tb_x * size_of::<f32>()).into(),
            outputs: vec![batch * seq * hidden],
        },
        (batch, seq, hidden),
        x.graph(),
    );
    y
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

    let out = custom_kernel(
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
        (batch, seq, hidden_dim),
        queries.graph(),
    );
    out
}

fn apply_rotary_embeddings_cuda(input: GraphTensor, prev_seq: Expression) -> GraphTensor {
    let (batch, seq, n_heads, head_dim) = input.dims4();
    let (seq_kernel, n_heads_kernel, head_dim_kernel, prev_seq_kernel) = (
        seq.to_kernel(),
        n_heads.to_kernel(),
        head_dim.to_kernel(),
        prev_seq.to_kernel(),
    );
    let x = custom_kernel(
        &[input],
        Kernel {
            code: format!(
                "
#include <math_constants.h>
#include <math_functions.h>

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ x,  // [B,S,H,D]
    float* __restrict__ y,        // [B,H,S,D]
    const size_t const_p, const size_t const_s
) {{
    // One block per (b,s,h); threads iterate over i in [0, D/2)
    int row = blockIdx.x;
    int s = row % {seq_kernel};
    int tmp = row / {seq_kernel};
    int h = tmp % {n_heads_kernel};
    int b = tmp / {n_heads_kernel};

    const float rope_base = 500000.0f;

    const int pairs = {head_dim_kernel} >> 1; // D/2

    // x layout: [B,S,H,D]
    size_t x_row = (((size_t)b * (size_t){seq_kernel} + (size_t)s) * (size_t){n_heads_kernel} + (size_t)h) * (size_t){head_dim_kernel};
    // y layout: [B,H,S,D]
    size_t y_row = (((size_t)b * (size_t){n_heads_kernel} + (size_t)h) * (size_t){seq_kernel} + (size_t)s) * (size_t){head_dim_kernel};

    float pos = (float)({prev_seq_kernel} + s);

    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {{
        size_t even_off = (size_t)(2 * i);
        size_t odd_off  = even_off + 1;

        float x0 = x[x_row + even_off];
        float x1 = x[x_row + odd_off];

        float inv   = 1.0f / powf(rope_base, (2.0f * i) / (float){head_dim_kernel});
        float angle = pos * inv;

        float sn, cs; __sincosf(angle, &sn, &cs);

        y[y_row + even_off] = x0 * cs - x1 * sn;
        y[y_row + odd_off ] = x0 * sn + x1 * cs;
    }}
}}"
            ),
            // one block per (b,s,h)
            grid: ((batch * seq * n_heads).into(), 1.into(), 1.into()),
            threadblock: (64.into(), 1.into(), 1.into()),
            smem: 0.into(),
            outputs: vec![input.shape.n_elements()],
        },
        // OUTPUT SHAPE is [B,H,S,D]
        (batch, n_heads, seq, head_dim),
        input.graph(),
    );
    x
}
fn proj_swiglu_cuda(x: GraphTensor, w_up: GraphTensor, w_gate: GraphTensor) -> GraphTensor {
    // x:[B,S,K], w_up:[N,K], w_gate:[N,K]  →  a:[B,S,N] with a = (X·W_up) ⊙ swish(X·W_gate)
    let (batch, seq, k) = x.dims3();
    let (n, _k2) = w_up.dims2();
    let (m_k, n_k, k_k) = ((batch * seq).to_kernel(), n.to_kernel(), k.to_kernel());

    let a = custom_kernel(
        &[x, w_up, w_gate],
        Kernel {
            code: format!(
                "
#include <math_functions.h>

// A[m,n] = (sum_k X[m,k]*W_up[n,k]) * swish(sum_k X[m,k]*W_gate[n,k])
// X logically [M,K] with M=B*S; W_up:[N,K]; W_gate:[N,K]; A:[M,N]
__inline__ __device__ float warp_sum(float v, unsigned mask){{
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}}

extern \"C\" __global__ void kernel_name(
    const float* __restrict__ X,     // [M,K] from [B,S,K]
    const float* __restrict__ Wup,   // [N,K]
    const float* __restrict__ Wgt,   // [N,K]
    float* __restrict__ A,           // [M,N]
    const size_t const_p, const size_t const_s)
{{
    const int M = {m_k};
    const int N = {n_k};
    const int K = {k_k};

    const int ncol = blockIdx.x;
    if (ncol >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned fullmask = __activemask();

    const float* __restrict__ wup = Wup + (size_t)ncol * K;
    const float* __restrict__ wgt = Wgt + (size_t)ncol * K;

    const int TM = 8;
    extern __shared__ float s[];
    float* red_u = s;                   // TM * nwarps
    float* red_g = s + TM * nwarps;     // TM * nwarps

    for (int m0 = 0; m0 < M; m0 += TM) {{
        int Me = min(TM, M - m0);

        float acc_u[TM], acc_g[TM];
        #pragma unroll
        for (int i=0;i<TM;++i){{ acc_u[i]=0.f; acc_g[i]=0.f; }}

        const int K4 = (K >> 2) << 2;
        for (int k4 = tid; k4 < (K4 >> 2); k4 += blockDim.x) {{
            int k = k4 << 2;
            float4 uv = *reinterpret_cast<const float4*>(wup + k);
            float4 gv = *reinterpret_cast<const float4*>(wgt + k);
            #pragma unroll
            for (int i=0;i<TM;++i){{
                if (i>=Me) break;
                const float4* __restrict__ x4 =
                    reinterpret_cast<const float4*>(X + (size_t)(m0+i)*K + k);
                float4 xv = *x4;
                acc_u[i] = fmaf(xv.x, uv.x, acc_u[i]); acc_g[i] = fmaf(xv.x, gv.x, acc_g[i]);
                acc_u[i] = fmaf(xv.y, uv.y, acc_u[i]); acc_g[i] = fmaf(xv.y, gv.y, acc_g[i]);
                acc_u[i] = fmaf(xv.z, uv.z, acc_u[i]); acc_g[i] = fmaf(xv.z, gv.z, acc_g[i]);
                acc_u[i] = fmaf(xv.w, uv.w, acc_u[i]); acc_g[i] = fmaf(xv.w, gv.w, acc_g[i]);
            }}
        }}
        for (int k = K4 + tid; k < K; k += blockDim.x) {{
            float wu = wup[k], wg = wgt[k];
            #pragma unroll
            for (int i=0;i<TM;++i){{
                if (i<Me) {{
                    float xv = X[(size_t)(m0+i)*K + k];
                    acc_u[i] = fmaf(xv, wu, acc_u[i]);
                    acc_g[i] = fmaf(xv, wg, acc_g[i]);
                }}
            }}
        }}

        #pragma unroll
        for (int i=0;i<TM;++i){{
            float vu = (i<Me)? acc_u[i]:0.f;
            float vg = (i<Me)? acc_g[i]:0.f;
            vu = warp_sum(vu, fullmask);
            vg = warp_sum(vg, fullmask);
            if (lane==0 && i<Me){{ red_u[i*nwarps + wid]=vu; red_g[i*nwarps + wid]=vg; }}
        }}
        __syncthreads();

        if (wid==0){{
            #pragma unroll
            for (int i=0;i<TM;++i){{
                if (i>=Me) break;
                float su = (lane < nwarps) ? red_u[i*nwarps + lane] : 0.f;
                float sg = (lane < nwarps) ? red_g[i*nwarps + lane] : 0.f;
                su = warp_sum(su, 0xffffffff);
                sg = warp_sum(sg, 0xffffffff);
                if (lane==0){{
                    float sig = 1.f / (1.f + expf(-sg));
                    A[(size_t)(m0+i)*N + ncol] = su * (sg * sig); // Swish
                }}
            }}
        }}
        __syncthreads();
    }}
}}"
            ),
            grid: (n.into(), 1.into(), 1.into()),
            threadblock: (256.into(), 1.into(), 1.into()),
            smem: (2 * 8 * ((256 + 31) / 32) * core::mem::size_of::<f32>()).into(),
            outputs: vec![batch * seq * n],
        },
        (batch, seq, n),
        x.graph(),
    );
    a
}
