//! moondream2.rs
//! Luminal implementation of Moondream 2 (vision‑language model)

use luminal::prelude::{binary::F32Pow, *};
#[allow(unused_imports)]
use luminal_nn::{Conv2D, Embedding, LayerNorm, Linear};

////////////////////////////////////////////////////////////////
// Model‑wide constants – taken from moondream/torch/config.py //
////////////////////////////////////////////////////////////////

// —— Text (“GPT”) tower ——
pub const TXT_DIM: usize = 2048;
pub const TXT_FF_DIM: usize = 8192;
pub const TXT_N_LAYERS: usize = 24;
pub const TXT_VOCAB: usize = 51_200;
#[allow(dead_code)]
pub const TXT_MAX_CTX: usize = 2048;
pub const TXT_N_HEADS: usize = 32;
pub const TXT_N_KV: usize = 32;

// —— Vision encoder ——
pub const VIS_DIM: usize = 1152;
pub const VIS_PATCH: usize = 14;
pub const VIS_LAYERS: usize = 27;
pub const VIS_FF_DIM: usize = 4304;
pub const VIS_HEADS: usize = 16;
pub const VIS_CROP: usize = 378;
#[allow(dead_code)]
pub const VIS_MAX_CROPS: usize = 12;
pub const VIS_PROJ_DIM: usize = 2048;
pub const VIS_PROJ_INNER: usize = 8192;

// —— Region head ——
pub const REG_DIM: usize = 2048;
pub const REG_COORD_FEAT: usize = 256;
pub const REG_COORD_OUT: usize = 1024;
pub const REG_SIZE_FEAT: usize = 512;
pub const REG_SIZE_OUT: usize = 2048;
pub const REG_INNER: usize = 8192;

//////////////////////////////////////////////////////
// Helper: rotary embedding for single‑precision FFT //
//////////////////////////////////////////////////////

fn apply_rotary_embeddings(t: GraphTensor, pos: Expression) -> GraphTensor {
    let (b, h, s, d) = t.dims4();
    let freqs = (t.graph().arange(d / 2) * 2.0) / (d.to_usize().unwrap() as f32);
    let freqs = 500_000_f32.pow(freqs); // θ_i = 500k^(-2i/d)
    let pos = t.graph().arange(s) + pos; // absolute positions
    let emb = pos.expand_dim(1, 1).matmul(freqs.expand_dim(0, 1));

    let split = t.reshape((b, h, s, d / 2, 2));
    let x0 = split.slice((.., .., .., .., ..1));
    let x1 = split.slice((.., .., .., .., 1..));

    let out0 = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
    let out1 = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

    out0.concat_along(out1, 4).reshape(t.shape)
}

/////////////////////////////
// Vision encoder (ViT‑E) //
/////////////////////////////

pub const PATCH_DIM: usize = VIS_PATCH * VIS_PATCH * 3; // 588
pub struct PatchEmbed {
    lin: Linear, // 588 → 1152
}
impl PatchEmbed {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            lin: Linear::new_permuted(PATCH_DIM, VIS_DIM, false, cx),
        }
    }
}
impl Module<GraphTensor> for PatchEmbed {
    type Output = GraphTensor; // (B,‑,1152)
    fn forward(&self, x: GraphTensor) -> GraphTensor {
        // x : (B,C,H,W)  with H=W=378
        let (b, c, h, w) = x.dims4();
        let p = VIS_PATCH; // 14
                           // Step‑1: B C H/P P W/P P
        let x = x
            .reshape((b, c, h / p, p, w / p, p))
            // Step‑2: B H/P W/P C P P
            .permute((0, 2, 4, 1, 3, 5))
            // Step‑3: B N 588   where N = 27×27 = 729
            .reshape((b, (h / p) * (w / p), PATCH_DIM));
        self.lin.forward(x) // (B,729,1152)
    }
}

pub struct ViTBlock {
    ln1: LayerNorm,
    ln2: LayerNorm,
    qkv: Linear,
    proj: Linear, // fused 3 × dim
    fc1: Linear,
    fc2: Linear,
}
impl ViTBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            ln1: LayerNorm::new(VIS_DIM, true, false, false, 1e-5, cx),
            ln2: LayerNorm::new(VIS_DIM, true, false, false, 1e-5, cx),
            qkv: Linear::new_permuted(VIS_DIM, 3 * VIS_DIM, false, cx),
            proj: Linear::new_permuted(VIS_DIM, VIS_DIM, false, cx),
            fc1: Linear::new_permuted(VIS_DIM, VIS_FF_DIM, false, cx),
            fc2: Linear::new_permuted(VIS_FF_DIM, VIS_DIM, false, cx),
        }
    }
}
impl Module<GraphTensor> for ViTBlock {
    type Output = GraphTensor;
    fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        let (b, n, _) = x.dims3();
        let qkv = self
            .qkv
            .forward(self.ln1.forward(x))
            .reshape((b, n, 3, VIS_HEADS, VIS_DIM / VIS_HEADS))
            .permute((2, 0, 3, 1, 4)); // 3 × B × H × N × d
        let (q, k, v) = (qkv.slice((0, ..)), qkv.slice((1, ..)), qkv.slice((2, ..)));
        let y = q.matmul(k.permute((0, 1, 2, 4, 3))) / (VIS_DIM as f32 / VIS_HEADS as f32).sqrt();
        let y = y
            .softmax(3)
            .matmul(v)
            .permute((1, 3, 0, 2, 4))
            .reshape((b, n, VIS_DIM));
        x += self.proj.forward(y);

        let z = self.fc1.forward(self.ln2.forward(x)).swish();
        x + self.fc2.forward(z)
    }
}

pub struct VisionEncoder {
    patch: PatchEmbed,
    pos: GraphTensor,
    blks: Vec<ViTBlock>,
}
impl VisionEncoder {
    pub fn new(cx: &mut Graph) -> Self {
        let patch = PatchEmbed::new(cx);
        let n_patches = (VIS_CROP / VIS_PATCH).pow(2);
        Self {
            patch,
            pos: cx.named_tensor("vis_pos", (1, n_patches + 1, VIS_DIM)), // +CLS
            blks: (0..VIS_LAYERS).map(|_| ViTBlock::new(cx)).collect(),
        }
    }
}
impl Module<GraphTensor> for VisionEncoder {
    type Output = GraphTensor; // (b, n_tokens, VIS_DIM)
    fn forward(&self, x: GraphTensor) -> Self::Output {
        let mut t = self.patch.forward(x);
        t = t.concat_along(self.pos.slice((.., ..1, ..)), 1); // prepend CLS
        t += self.pos.slice((.., 1.., ..)); // add patch pos
        for blk in &self.blks {
            t = blk.forward(t);
        }
        t
    }
}

//////////////////////////////////////////
// Simple projection to Text prefix     //
//////////////////////////////////////////

pub struct VisionProjection {
    fc1: Linear,
    fc2: Linear,
}
impl VisionProjection {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(VIS_DIM * 2, VIS_PROJ_INNER, false, cx),
            fc2: Linear::new_permuted(VIS_PROJ_INNER, VIS_PROJ_DIM, false, cx),
        }
    }
}
impl Module<GraphTensor> for VisionProjection {
    type Output = GraphTensor;
    fn forward(&self, vis: GraphTensor) -> GraphTensor {
        let (b, n, _) = vis.dims3(); // n == 729
        let g = vis.slice((.., ..1, ..)).expand((b, n, VIS_DIM));
        let grid = vis.slice((.., 1.., ..));
        let feats = g.concat_along(grid, 2); // (b,729,2304)
        self.fc2.forward(self.fc1.forward(feats).swish())
    }
}

//////////////////////////
// Region MLP head      //
//////////////////////////

pub struct RegionHead {
    enc_coord: Linear,
    enc_size: Linear,
    dec_coord: (Linear, Linear),
    dec_size: (Linear, Linear),
    coord_feat: GraphTensor,
    size_feat: GraphTensor,
}
impl RegionHead {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            enc_coord: Linear::new_permuted(REG_COORD_FEAT, REG_DIM, false, cx),
            enc_size: Linear::new_permuted(REG_SIZE_FEAT, REG_DIM, false, cx),
            dec_coord: (
                Linear::new_permuted(REG_DIM, REG_INNER, false, cx),
                Linear::new_permuted(REG_INNER, REG_COORD_OUT, false, cx),
            ),
            dec_size: (
                Linear::new_permuted(REG_DIM, REG_INNER, false, cx),
                Linear::new_permuted(REG_INNER, REG_SIZE_OUT, false, cx),
            ),
            coord_feat: cx.named_tensor("coord_feat", (REG_COORD_FEAT / 2, 1)),
            size_feat: cx.named_tensor("size_feat", (REG_SIZE_FEAT / 2, 2)),
        }
    }
}

/////////////////////////////////////////
// Text transformer – identical style  //
/////////////////////////////////////////

#[allow(dead_code)]
pub const TXT_ATT_GROUPS: usize = 1; // n_heads == n_kv_heads
pub const TXT_HEAD_DIM: usize = TXT_DIM / TXT_N_HEADS;

pub type KVCache = (GraphTensor, GraphTensor);

// ─────────────────── Self‑Attention with fused QKV ────────────────────
pub struct SelfAttention {
    qkv: Linear,  // TXT_DIM → (n_heads + 2·n_kv) · head_dim
    proj: Linear, // TXT_DIM → TXT_DIM
}

impl SelfAttention {
    pub fn new(cx: &mut Graph) -> Self {
        //  q_dim = H·d   kv_dim = 2·Hkv·d
        const QKV_DIM: usize = TXT_DIM * (1 + 2 * TXT_N_KV / TXT_N_HEADS); // 2048 * (1+2) = 6144
        Self {
            qkv: Linear::new_permuted(TXT_DIM, QKV_DIM, false, cx),
            proj: Linear::new_permuted(TXT_DIM, TXT_DIM, false, cx),
        }
    }
}

impl Module<(GraphTensor, KVCache)> for SelfAttention {
    type Output = (GraphTensor, KVCache);

    fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
        let (b, s, _) = x.dims3();
        let (_, _, p, _) = k_cache.dims4();
        let head_dim = TXT_HEAD_DIM;

        // fused projection
        let qkv = self
            .qkv
            .forward(x)
            .reshape((b, s, TXT_N_HEADS + 2 * TXT_N_KV, head_dim)); // (B,S,H+2Hkv,d)

        let q = qkv
            .slice((.., .., 0..TXT_N_HEADS, ..))
            .permute((0, 2, 1, 3)); // (B,H,S,d)

        let k = qkv
            .slice((.., .., TXT_N_HEADS..TXT_N_HEADS + TXT_N_KV, ..))
            .permute((0, 2, 1, 3)); // (B,Hkv,S,d)

        let v = qkv
            .slice((.., .., TXT_N_HEADS + TXT_N_KV.., ..))
            .permute((0, 2, 1, 3)); // (B,Hkv,S,d)

        // rotary & cache
        let q = apply_rotary_embeddings(q, p);
        let k = apply_rotary_embeddings(k, p);
        let k = k_cache.concat_along(k, 2);
        let v = v_cache.concat_along(v, 2);

        // attention
        let att = q.matmul(k.permute((0, 1, 3, 2))) / (head_dim as f32).sqrt();
        let mask = self.qkv.weight.graph().triu(s, 1) * f16::MIN.to_f32();
        let att = (att
            + mask
                .pad(((0, 0), (p, 0)))
                .expand_dim(0, b)
                .expand_dim(1, TXT_N_KV))
        .softmax(3);

        let out = att
            .matmul(v)
            .permute((0, 2, 1, 3)) // (B,S,H,d)
            .reshape((b, s, TXT_DIM));

        (self.proj.forward(out), (k.contiguous(), v.contiguous()))
    }
}

// ─────────────── Serialization (weight names) ───────────────
impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.module("qkv", &self.qkv); // weight key: ".../qkv"
        s.module("proj", &self.proj); // weight key: ".../proj"
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct FFN {
    fc1: Linear,
    fc2: Linear,
}
impl FFN {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(TXT_DIM, TXT_FF_DIM, false, cx),
            fc2: Linear::new_permuted(TXT_FF_DIM, TXT_DIM, false, cx),
        }
    }
}
impl Module<GraphTensor> for FFN {
    type Output = GraphTensor;
    fn forward(&self, x: GraphTensor) -> Self::Output {
        self.fc2.forward(self.fc1.forward(x).gelu())
    }
}

pub struct TextBlock {
    ln: LayerNorm,
    attn: SelfAttention,
    ffn: FFN,
}
impl TextBlock {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            ln: LayerNorm::new(TXT_DIM, true, false, false, 1e-5, cx),
            attn: SelfAttention::new(cx),
            ffn: FFN::new(cx),
        }
    }
}
impl Module<(GraphTensor, KVCache)> for TextBlock {
    type Output = (GraphTensor, KVCache);
    fn forward(&self, (x, cache): (GraphTensor, KVCache)) -> Self::Output {
        let normed = self.ln.forward(x);
        let (y, cache) = self.attn.forward((normed, cache));
        let z = self.ffn.forward(normed);
        (x + y + z, cache)
    }
}

//////////////////////////////////
// Full Moondream 2 top‑level   //
//////////////////////////////////

pub struct Moondream {
    vision: VisionEncoder,
    vis_proj: VisionProjection,
    region: RegionHead,
    embed: Embedding,
    txt_blocks: Vec<TextBlock>,
    txt_norm: LayerNorm,
    lm_head: Linear,
}
impl Moondream {
    pub fn new(cx: &mut Graph) -> Self {
        Self {
            vision: VisionEncoder::new(cx),
            vis_proj: VisionProjection::new(cx),
            region: RegionHead::new(cx),
            embed: Embedding::new(TXT_VOCAB, TXT_DIM, cx),
            txt_blocks: (0..TXT_N_LAYERS).map(|_| TextBlock::new(cx)).collect(),
            txt_norm: LayerNorm::new(TXT_DIM, true, false, false, 1e-5, cx),
            lm_head: Linear::new_permuted(TXT_DIM, TXT_VOCAB, false, cx),
        }
    }
}

impl Module<(GraphTensor, GraphTensor, &[KVCache])> for Moondream {
    // Args: (image_tensor, token_ids, kv_cache[])
    type Output = (GraphTensor, Vec<KVCache>);
    fn forward(&self, (img, toks, cache): (GraphTensor, GraphTensor, &[KVCache])) -> Self::Output {
        // Vision → prefix tokens
        let vis_tokens = self.vision.forward(img); // (b,n_vis,VIS_DIM)
        let prefix = self.vis_proj.forward(vis_tokens); // (b,prefix,TXT_DIM)

        // Text embedding
        let mut x = self.embed.forward(toks); // (b,seq,TXT_DIM)
        x = prefix.concat_along(x, 1); // prepend vision prefix

        // Transformer
        let mut new = Vec::with_capacity(TXT_N_LAYERS);
        for (blk, i) in self.txt_blocks.iter().zip(0..) {
            let (y, c) = blk.forward((x, cache[i]));
            x = y;
            new.push(c);
        }

        (self.lm_head.forward(self.txt_norm.forward(x)), new)
    }
}

//////////////////////////////////////////////////////
// 🔹  Serialization helpers for Moondream 2
//////////////////////////////////////////////////////

impl SerializeModule for ViTBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln1", &self.ln1);
        s.module("attn/qkv", &self.qkv);
        s.module("attn/proj", &self.proj);
        s.module("ln2", &self.ln2);
        s.module("mlp/fc1", &self.fc1);
        s.module("mlp/fc2", &self.fc2);
    }
}

impl SerializeModule for VisionEncoder {
    fn serialize(&self, s: &mut Serializer) {
        s.module("patch_emb", &self.patch.lin);
        s.tensor("pos_emb", self.pos);
        for (i, b) in self.blks.iter().enumerate() {
            s.module(&format!("blocks/{i}"), b);
        }
        // post‑ln is already inside VisionProjection MLP (proj_mlp)
    }
}

impl SerializeModule for VisionProjection {
    fn serialize(&self, s: &mut Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
    }
}

impl SerializeModule for RegionHead {
    fn serialize(&self, s: &mut Serializer) {
        s.module("coord_encoder", &self.enc_coord);
        s.module("coord_decoder/fc1", &self.dec_coord.0);
        s.module("coord_decoder/fc2", &self.dec_coord.1);
        s.module("size_encoder", &self.enc_size);
        s.module("size_decoder/fc1", &self.dec_size.0);
        s.module("size_decoder/fc2", &self.dec_size.1);
        s.tensor("coord_features", self.coord_feat);
        s.tensor("size_features", self.size_feat);
    }
}

// impl SerializeModule for SelfAttention {
//     fn serialize(&self, s: &mut Serializer) {
//         s.tensor("q_proj/weight", self.q_proj);
//         s.tensor("k_proj/weight", self.k_proj);
//         s.tensor("v_proj/weight", self.v_proj);
//         s.tensor("o_proj/weight", self.o_proj);
//     }
// }

impl SerializeModule for FFN {
    fn serialize(&self, s: &mut Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
    }
}

impl SerializeModule for TextBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("ln", &self.ln);
        s.module("attn", &self.attn);
        s.module("mlp", &self.ffn);
    }
}

impl SerializeModule for Moondream {
    fn serialize(&self, s: &mut Serializer) {
        // Vision branch
        s.module("model/vision", &self.vision);
        s.module("model/vision/proj_mlp", &self.vis_proj);
        s.module("model/region", &self.region);

        // Text branch
        s.tensor("model/text/wte", self.embed.weight);
        for (i, blk) in self.txt_blocks.iter().enumerate() {
            s.module(&format!("model/text/blocks/{i}"), blk);
        }
        s.module("model/text/post_ln", &self.txt_norm);
        s.module("model/text/lm_head", &self.lm_head);
    }
}
