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

        //         let [output] = custom_kernel(
        //             &[queries, keys, values],
        //             Kernel {
        //                 code: format!(
        //                     "
        // #define TILE_Q 32
        // #define TILE_K 64
        // #define HEAD_DIM_MAX 128

        // extern \"C\" __global__ void kernel_name(
        //     const float* __restrict__ Q,   // [B, N_KV_HEADS, {N_ATTENTION_GROUPS}, const_s, {HEAD_DIM}]
        //     const float* __restrict__ K,   // [B, N_KV_HEADS, const_s + const_p, {HEAD_DIM}]
        //     const float* __restrict__ V,   // [B, N_KV_HEADS, const_s + const_p, {HEAD_DIM}]
        //     float* __restrict__ Out,       // [B, N_KV_HEADS, {N_ATTENTION_GROUPS}, const_s, {HEAD_DIM}]
        //     const size_t const_p,
        //     const size_t const_s
        // ){{
        //     const int b  = blockIdx.x;
        //     const int hg = blockIdx.y;                  // packs (g, N_KV_HEADS)
        //     const int g  = hg / {N_KV_HEADS};
        //     const int h  = hg % {N_KV_HEADS};
        //     const int q_tile = blockIdx.z * TILE_Q;

        //     const int tid = threadIdx.x;                // 0..TILE_Q-1
        //     if (g >= {N_ATTENTION_GROUPS} || h >= {N_KV_HEADS} || tid >= TILE_Q) return;

        //     const int qi = q_tile + tid;                // query index in this tile
        //     if (qi >= const_s) return;

        //     // Base pointers (row-major)
        //     const size_t strideQ_b = (size_t){N_KV_HEADS} * {N_ATTENTION_GROUPS} * const_s * {HEAD_DIM};
        //     const size_t strideQ_h = (size_t){N_ATTENTION_GROUPS} * const_s * {HEAD_DIM};
        //     const size_t strideQ_g = (size_t)const_s * {HEAD_DIM};

        //     const size_t strideK_b = (size_t){N_KV_HEADS} * const_s + const_p * {HEAD_DIM};
        //     const size_t strideK_h = (size_t)const_s + const_p * {HEAD_DIM};

        //     const size_t strideO_b = (size_t){N_KV_HEADS} * {N_ATTENTION_GROUPS} * const_s * {HEAD_DIM};
        //     const size_t strideO_h = (size_t){N_ATTENTION_GROUPS} * const_s * {HEAD_DIM};
        //     const size_t strideO_g = (size_t)const_s * {HEAD_DIM};

        //     const float* Qptr = Q + b*strideQ_b + h*strideQ_h + g*strideQ_g + (size_t)qi*{HEAD_DIM};
        //     const float* Kbase = K + b*strideK_b + h*strideK_h;
        //     const float* Vbase = V + b*strideK_b + h*strideK_h;
        //     float* Outptr = Out + b*strideO_b + h*strideO_h + g*strideO_g + (size_t)qi*{HEAD_DIM};

        //     // Load this query row into registers
        //     float q_row[HEAD_DIM_MAX];
        //     for (int d = 0; d < {HEAD_DIM}; ++d) q_row[d] = Qptr[d];

        //     // Online softmax state for this query row
        //     const float scale = rsqrtf((float){HEAD_DIM});
        //     float m_i = -__int_as_float(0x7f800000);   // running max
        //     float l_i = 0.f;             // running sum of exp
        //     float out_i[HEAD_DIM_MAX];   // accumulated output
        //     for (int d = 0; d < {HEAD_DIM}; ++d) out_i[d] = 0.f;

        //     // Absolute query position (for causal maconst_s + const_ping)
        //     const int q_abs = const_p + qi;

        //     extern __shared__ float smem[]; // size = (TILE_K*{HEAD_DIM})*2 floats
        //     float* Ks = smem;               // [TILE_K, {HEAD_DIM}]
        //     float* Vs = smem + (size_t)TILE_K * {HEAD_DIM};

        //     // Iterate over keys in tiles of TILE_K
        //     for (int k0 = 0; k0 < const_s + const_p; k0 += TILE_K) {{
        //         const int tileK = min((int)TILE_K, (int)(const_s + const_p - k0));

        //         // Cooperative load: K and V tiles into shared
        //         // Flattened idx over tileK*{HEAD_DIM}
        //         for (int t = tid; t < tileK*{HEAD_DIM}; t += blockDim.x) {{
        //             int tr = t / {HEAD_DIM};   // 0..tileK-1  (key row)
        //             int tc = t % {HEAD_DIM};   // 0..{HEAD_DIM}-1
        //             Ks[tr*{HEAD_DIM} + tc] = Kbase[(k0 + tr)*(size_t){HEAD_DIM} + tc];
        //             Vs[tr*{HEAD_DIM} + tc] = Vbase[(k0 + tr)*(size_t){HEAD_DIM} + tc];
        //         }}
        //         __syncthreads();

        //         // Compute current tile logits for this query row and online update
        //         // First, find max within this tile (with causal maconst_s + const_p)
        //         float m_tile = -__int_as_float(0x7f800000);
        //         // We’ll also cache logits to registers to avoid recomputing dot(Q,K)
        //         // (bounded by TILE_K)
        //         float logits[TILE_K];

        //         for (int tk = 0; tk < tileK; ++tk) {{
        //             const int k_abs = k0 + tk;
        //             float s = -__int_as_float(0x7f800000);
        //             if (k_abs <= q_abs) {{  // causal maconst_s + const_p
        //                 // dot(q_row, Ks[tk,:])
        //                 float dot = 0.f;
        //                 const float* __restrict__ Krow = &Ks[tk*{HEAD_DIM}];
        //                 // Unrolled by 4 is often fine; keep simple here
        //                 for (int d = 0; d < {HEAD_DIM}; ++d) dot += q_row[d] * Krow[d];
        //                 s = dot * scale;
        //             }}
        //             logits[tk] = s;
        //             m_tile = fmaxf(m_tile, s);
        //         }}

        //         // Merge max with running max
        //         const float m_new = fmaxf(m_i, m_tile);

        //         // Compute exp(logits - m_new), accumulate l and out
        //         // Also apply rescaling factor to previous accumulators.
        //         float l_tile = 0.f;
        //         const float alpha = __expf(m_i - m_new);   // rescales previous terms
        //         for (int d = 0; d < {HEAD_DIM}; ++d) out_i[d] *= alpha;

        //         for (int tk = 0; tk < tileK; ++tk) {{
        //             float p = __expf(logits[tk] - m_new);  // 0 if maconst_s + const_ped
        //             l_tile += p;
        //             const float* __restrict__ Vrow = &Vs[tk*{HEAD_DIM}];
        //             for (int d = 0; d < {HEAD_DIM}; ++d) out_i[d] += p * Vrow[d];
        //         }}

        //         l_i = l_i * alpha + l_tile;
        //         m_i = m_new;

        //         __syncthreads();
        //     }}

        //     // Normalize
        //     const float inv_l = 1.f / fmaxf(l_i, 1e-20f);
        //     for (int d = 0; d < {HEAD_DIM}; ++d) Outptr[d] = out_i[d] * inv_l;
        // }}"
        //                 ),
        //                 grid: (1.into(), HEAD_DIM.into(), ((seq + 32 - 1) / 32).into()),
        //                 threadblock: (32.into(), 1.into(), 1.into()),
        //                 smem: (64 * HEAD_DIM * 2 * size_of::<f32>()).into(),
        //                 outputs: vec![batch * seq * N_KV_HEADS * N_ATTENTION_GROUPS * HIDDEN_DIM],
        //             },
        //             [(batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HIDDEN_DIM)],
        //             queries.graph(),
        //         );

        // Repeat the KV States for Grouped-Query Attention
        let repeated_keys = keys.expand_dim(2, N_ATTENTION_GROUPS);
        let repeated_values = values.expand_dim(2, N_ATTENTION_GROUPS);

        let mut attention_weights = queries
            .reshape((batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM))
            .matmul(repeated_keys.permute((0, 1, 2, 4, 3)))
            / (HEAD_DIM as f32).sqrt();

        // Causal maconst_s + const_p
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
            .matmul(repeated_values);

        let output = output
            // Merge heads
            .permute((0, 3, 1, 2, 4))
            .reshape((batch, seq, HIDDEN_DIM));
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
