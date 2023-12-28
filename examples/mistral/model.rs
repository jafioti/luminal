use luminal::{
    nn::{embedding::Embedding, norm::RMSNorm},
    prelude::*,
    shape::symbolic::Expression,
};

//////////////////////////////////////////////
///          Mistral 7B Config             ///
//////////////////////////////////////////////

pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
// pub const NUM_LAYERS: usize = 1;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const MLP_PROJECTION_DIM: usize = 14336;
pub const ROPE_THETA: f32 = 1000000.0;
pub const MAX_POSITION_EMBEDDINGS: usize = 4096;

pub const NUM_ATTENTION_GROUPS: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS;
pub const ATTENTION_HEAD_DIM: usize = HIDDEN_DIM / NUM_ATTENTION_HEADS;
pub const ATTENTION_HEAD_DIM_OVER_2: usize = ATTENTION_HEAD_DIM / 2;
pub const ATTENTION_PROJECTION_DIM: usize = ATTENTION_HEAD_DIM * NUM_KV_HEADS;

//////////////////////////////////////////////
///              Model Structs             ///
//////////////////////////////////////////////

pub struct Mlp<const I: usize, const H: usize> {
    pub gate_proj: GraphTensor<(Const<I>, Const<H>)>,
    pub down_proj: GraphTensor<(Const<H>, Const<I>)>,
    pub up_proj: GraphTensor<(Const<I>, Const<H>)>,
}

impl<Sh: Shape, Im: Shape, const I: usize, const H: usize> Module<GraphTensor<Sh>> for Mlp<I, H>
where
    GraphTensor<Sh>: Matmul<R2<H, I>, Output = GraphTensor<Im>>,
    GraphTensor<Im>: Matmul<R2<I, H>, Output = GraphTensor<Sh>>,
{
    type Output = GraphTensor<Sh>;

    fn forward(&self, input: GraphTensor<Sh>) -> Self::Output {
        let gate = input.matmul(self.gate_proj.permute()).swish();
        let up = input.matmul(self.up_proj.permute()) * gate;
        up.matmul(self.down_proj.permute())
    }
}

impl<const I: usize, const H: usize> InitModule for Mlp<I, H> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            gate_proj: cx.named_tensor("Gate Weight"),
            up_proj: cx.named_tensor("Up Weight"),
            down_proj: cx.named_tensor("Down Weight"),
        }
    }
}

impl<const I: usize, const H: usize> SerializeModule for Mlp<I, H> {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("gate_proj/weight", self.gate_proj);
        s.tensor("up_proj/weight", self.up_proj);
        s.tensor("down_proj/weight", self.down_proj);
    }
}

pub struct SelfAttention {
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub k_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<ATTENTION_PROJECTION_DIM, HIDDEN_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
}

impl<Batch: Dimension, SequenceLength: Dimension>
    Module<(
        GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
        RotaryEmbeddings<SequenceLength>,
    )> for SelfAttention
{
    type Output = GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>;
    fn forward(
        &self,
        (x, rotary_embeddings): (
            GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
            RotaryEmbeddings<SequenceLength>,
        ),
    ) -> Self::Output {
        // Apply the Projections
        let query_states = x
            .matmul(self.q_proj.permute())
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let key_states = x
            .matmul(self.k_proj.permute())
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_KV_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let value_states = x
            .matmul(self.v_proj.permute())
            .reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_KV_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // Apply the Rotary Embeddings
        let query_states =
            apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(query_states, rotary_embeddings);

        let key_states =
            apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(key_states, rotary_embeddings);

        // Repeat the KV States for Grouped-Query Attention
        let key_states = key_states
            .expand::<(_, _, Const<NUM_ATTENTION_GROUPS>, _, _), Axis<2>>()
            .reshape::<(
                Batch,
                Const<NUM_ATTENTION_HEADS>,
                SequenceLength,
                Const<ATTENTION_HEAD_DIM>,
            )>()
            .permute::<_, Axes4<0, 1, 3, 2>>();

        let value_states = value_states
            .expand::<(_, _, Const<NUM_ATTENTION_GROUPS>, _, _), Axis<2>>()
            .reshape::<(
                Batch,
                Const<NUM_ATTENTION_HEADS>,
                SequenceLength,
                Const<ATTENTION_HEAD_DIM>,
            )>();

        let attention_weights = query_states.matmul(key_states);
        let attention_weights = attention_weights * (ATTENTION_HEAD_DIM as f32).sqrt().recip();

        let attention_weights = attention_weights.softmax::<3>();

        attention_weights
            .matmul(value_states)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Batch, SequenceLength, Const<HIDDEN_DIM>)>()
            .matmul(self.o_proj.permute())
    }
}

impl InitModule for SelfAttention {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            q_proj: cx.named_tensor("Q Proj"),
            k_proj: cx.named_tensor("K Proj"),
            v_proj: cx.named_tensor("V Proj"),
            o_proj: cx.named_tensor("O Proj"),
        }
    }
}

impl SerializeModule for SelfAttention {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("q_proj/weight", self.q_proj);
        s.tensor("v_proj/weight", self.v_proj);
        s.tensor("k_proj/weight", self.k_proj);
        s.tensor("o_proj/weight", self.o_proj);
    }
}

pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub attention_norm: RMSNorm<HIDDEN_DIM>,
    pub feed_forward: Mlp<MLP_PROJECTION_DIM, HIDDEN_DIM>,
    pub feed_forward_norm: RMSNorm<HIDDEN_DIM>,
}

impl<Batch: Dimension, SequenceLength: Dimension>
    Module<(
        GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
        RotaryEmbeddings<SequenceLength>,
    )> for TransformerBlock
{
    type Output = GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>;
    fn forward(
        &self,
        (x, rotary_embeddings): (
            GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
            RotaryEmbeddings<SequenceLength>,
        ),
    ) -> Self::Output {
        // Attention
        let mut residual = x;
        let mut x = self.attention_norm.forward(x);
        x = self.attention.forward((x, rotary_embeddings));

        // Residual Addition
        x += residual;

        // Feed Forward
        residual = x;
        x = self.feed_forward_norm.forward(x);
        x = self.feed_forward.forward(x);

        // Residual Addition
        x += residual;

        // Return
        x
    }
}

impl InitModule for TransformerBlock {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            attention_norm: {
                let mut norm = RMSNorm::initialize(cx);
                norm.epsilon = 1e-5;
                norm
            },
            feed_forward: InitModule::initialize(cx),
            feed_forward_norm: {
                let mut norm = RMSNorm::initialize(cx);
                norm.epsilon = 1e-5;
                norm
            },
        }
    }
}

impl SerializeModule for TransformerBlock {
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("input_layernorm", &self.attention_norm);
        s.module("post_attention_layernorm", &self.feed_forward_norm);
        s.module("mlp", &self.feed_forward);
    }
}

pub type RotaryEmbeddings<SequenceLength> = (
    GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
    GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
);

pub struct MistralLM {
    // Token embeddings
    pub embedding: Embedding<VOCAB_SIZE, HIDDEN_DIM>,
    // Transformer layers
    pub layers: Vec<TransformerBlock>,
    // Final Norm layer
    pub norm: RMSNorm<HIDDEN_DIM>,
    // LM Head Layer
    pub lm_head: GraphTensor<R2<VOCAB_SIZE, HIDDEN_DIM>>,
    // RoPE Embeddings
    pub rotary_embeddings: RotaryEmbeddings<Const<MAX_POSITION_EMBEDDINGS>>,
}

impl<Batch: Dimension, Seq: Dimension> Module<GraphTensor<(Batch, Seq)>> for MistralLM {
    type Output = GraphTensor<(Batch, Seq, Const<VOCAB_SIZE>)>;
    fn forward(&self, input: GraphTensor<(Batch, Seq)>) -> Self::Output {
        let mut hidden_states = self.embedding.forward(input);
        // Extract the Rotary Embeddings
        let (cos, sin) = self.rotary_embeddings;
        let cos = cos.slice((..Seq::const_size(), ..)).contiguous().realize();
        let sin = sin.slice((..Seq::const_size(), ..)).contiguous().realize();

        // Now, we loop over all layers
        for layer in &self.layers {
            hidden_states = layer.forward((hidden_states, (cos, sin)));
        }

        // Finally, we call the final norm
        hidden_states = self.norm.forward(hidden_states);

        hidden_states.matmul(self.lm_head.permute())
    }
}

//////////////////////////////////////////////
///          Batch Forward Impls           ///
//////////////////////////////////////////////

//////////////////////////////////////////////
///        KV-Cache Forward Impls          ///
//////////////////////////////////////////////

// pub struct KVCache<Batch: Dimension, SequenceLength: Dimension> {
//     pub keys: GraphTensor<(
//         Batch,
//         Const<NUM_KV_HEADS>,
//         SequenceLength,
//         Const<ATTENTION_HEAD_DIM>,
//     )>,
//     pub values: GraphTensor<(
//         Batch,
//         Const<NUM_KV_HEADS>,
//         SequenceLength,
//         Const<ATTENTION_HEAD_DIM>,
//     )>,
// }

// impl SelfAttention {
//     fn forward_kv<
//         Batch: Dimension,
//         OutputSequenceLength: Dimension,
//         PreviousSequenceLength: Dimension,
//     >(
//         &self,
//         x: GraphTensor<(Batch, Const<1>, Const<HIDDEN_DIM>)>,
//         rotary_embeddings: RotaryEmbeddings<Const<1>>,
//         kv_cache: KVCache<Batch, PreviousSequenceLength>,
//     ) -> (
//         GraphTensor<(Batch, OutputSequenceLength, Const<HIDDEN_DIM>)>,
//         KVCache<Batch, OutputSequenceLength>,
//     ) {
//         // Apply the Projections
//         let query_states = x
//             .matmul(self.q_proj.permute())
//             .reshape::<(
//                 Batch,
//                 Const<1>,
//                 Const<NUM_ATTENTION_HEADS>,
//                 Const<ATTENTION_HEAD_DIM>,
//             )>()
//             .permute::<_, Axes4<0, 2, 1, 3>>();

//         let key_states = x
//             .matmul(self.k_proj.permute())
//             .reshape::<(
//                 Batch,
//                 Const<1>,
//                 Const<NUM_KV_HEADS>,
//                 Const<ATTENTION_HEAD_DIM>,
//             )>()
//             .permute::<_, Axes4<0, 2, 1, 3>>();
//         let value_states = x
//             .matmul(self.v_proj.permute())
//             .reshape::<(
//                 Batch,
//                 Const<1>,
//                 Const<NUM_KV_HEADS>,
//                 Const<ATTENTION_HEAD_DIM>,
//             )>()
//             .permute::<_, Axes4<0, 2, 1, 3>>();

//         // Apply the Rotary Embeddings
//         let query_states =
//             apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(query_states, rotary_embeddings);

//         let key_states =
//             apply_rotary_embeddings::<_, _, _, ATTENTION_HEAD_DIM>(key_states, rotary_embeddings);

//         // Now we append the KV-Cache
//         let key_states = kv_cache.keys.concat_along::<(
//             Batch,
//             Const<NUM_KV_HEADS>,
//             OutputSequenceLength,
//             Const<ATTENTION_HEAD_DIM>,
//         ), Axis<2>, _>(key_states);

//         let value_states = kv_cache.values.concat_along::<(
//             Batch,
//             Const<NUM_KV_HEADS>,
//             OutputSequenceLength,
//             Const<ATTENTION_HEAD_DIM>,
//         ), Axis<2>, _>(value_states);

//         let kv_cache = KVCache {
//             keys: key_states,
//             values: value_states,
//         };

//         // Repeat the KV States for Grouped-Query Attention
//         let key_states = key_states
//             .expand::<(
//                 Batch,
//                 Const<NUM_KV_HEADS>,
//                 Const<NUM_ATTENTION_GROUPS>,
//                 OutputSequenceLength,
//                 Const<ATTENTION_HEAD_DIM>,
//             ), Axis<2>>()
//             .reshape::<(
//                 Batch,
//                 Const<NUM_ATTENTION_HEADS>,
//                 OutputSequenceLength,
//                 Const<ATTENTION_HEAD_DIM>,
//             )>()
//             .permute::<_, Axes4<0, 1, 3, 2>>();

//         let value_states = value_states
//             .expand::<(
//                 Batch,
//                 Const<NUM_KV_HEADS>,
//                 Const<NUM_ATTENTION_GROUPS>,
//                 OutputSequenceLength,
//                 Const<ATTENTION_HEAD_DIM>,
//             ), Axis<2>>()
//             .reshape::<(
//                 Batch,
//                 Const<NUM_ATTENTION_HEADS>,
//                 OutputSequenceLength,
//                 Const<ATTENTION_HEAD_DIM>,
//             )>();

//         let scores = query_states.matmul(key_states);
//         let scores = scores * (ATTENTION_HEAD_DIM as f32).sqrt().recip();
//         let scores = scores.softmax::<3>();

//         let output = scores
//             .matmul(value_states)
//             .permute::<_, Axes4<0, 2, 1, 3>>()
//             .reshape::<(Batch, OutputSequenceLength, Const<HIDDEN_DIM>)>()
//             .matmul(self.o_proj.permute());

//         // Return the output along with the new kv-cache
//         (output, kv_cache)
//     }
// }

// impl TransformerBlock {
//     fn forward_kv<Batch: Dimension>()
// }

//////////////////////////////////////////////
///          Serialization Impls           ///
//////////////////////////////////////////////

impl InitModule for MistralLM {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            embedding: InitModule::initialize(cx),
            norm: InitModule::initialize(cx),
            lm_head: cx.named_tensor("LM Head"),
            rotary_embeddings: {
                let (cos, sin) = compute_rotary_embedding_frequencies();
                (
                    cx.named_tensor("Rope Embeddings Cos").set(cos).keep(),
                    cx.named_tensor("Rope Embeddings Sin").set(sin).keep(),
                )
            },
            layers: (0..NUM_LAYERS)
                .map(|_| InitModule::initialize(cx))
                .collect(),
        }
    }
}

impl SerializeModule for MistralLM {
    fn serialize(&self, s: &mut Serializer) {
        s.module("model/embed_tokens", &self.embedding);
        s.module("model/norm", &self.norm);
        s.tensor("lm_head/weight", self.lm_head);
        for (i, layer) in self.layers.iter().enumerate() {
            s.module(&format!("model/layers/{i}"), layer);
        }
    }
}

//////////////////////////////////////////////
///          Initialization Impls          ///
//////////////////////////////////////////////

//////////////////////////////////////////////
///            Helper Functions            ///
//////////////////////////////////////////////

// From examples/llama

// Rotary Embeddings
pub fn compute_rotary_embedding_frequencies() -> (Vec<f32>, Vec<f32>) {
    let mut rope_graph = Graph::new();

    let frequencies = (rope_graph.arange::<Const<ATTENTION_HEAD_DIM_OVER_2>>() * 2.0)
        / (ATTENTION_HEAD_DIM as f32);

    let frequencies = frequencies
        .pow2(ROPE_THETA)
        .recip()
        .reshape::<R2<1, ATTENTION_HEAD_DIM_OVER_2>>();
    let frequencies: GraphTensor<R2<1, ATTENTION_HEAD_DIM>> =
        frequencies.concat_along::<_, Axis<1>, _>(frequencies);
    let t = rope_graph
        .arange::<Const<MAX_POSITION_EMBEDDINGS>>()
        .reshape::<(Const<MAX_POSITION_EMBEDDINGS>, Const<1>)>();
    let frequencies = t.matmul(frequencies);

    let cos = frequencies.cos().retrieve();
    let sin = frequencies.sin().retrieve();

    rope_graph.compile(<(PreGenericCompiler, MetalFp32Compiler, PostGenericCompiler)>::default());

    rope_graph.execute();
    (cos.data(), sin.data())
}

pub fn rotate_half<
    Batch: Dimension,
    SequenceLength: Dimension,
    NumAttentionHeads: Dimension,
    const ATTENTION_HEAD_DIM: usize,
>(
    x: GraphTensor<(
        Batch,
        SequenceLength,
        NumAttentionHeads,
        Const<ATTENTION_HEAD_DIM>,
    )>,
) -> GraphTensor<(
    Batch,
    SequenceLength,
    NumAttentionHeads,
    Const<ATTENTION_HEAD_DIM>,
)> {
    let x1 = x
        .slice((.., .., .., ..Expression::from(ATTENTION_HEAD_DIM / 2)))
        .contiguous();
    let x2 = x
        .slice((.., .., .., Expression::from(ATTENTION_HEAD_DIM / 2)..))
        .contiguous();

    (-x2).concat_along::<_, Axis<3>, _>(x1)
}

#[allow(clippy::type_complexity)]
pub fn apply_rotary_embeddings<
    Batch: Dimension,
    SequenceLength: Dimension,
    NumAttentionHeads: Dimension,
    const ATTENTION_HEAD_DIM: usize,
>(
    input: GraphTensor<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM>,
    )>,
    (cos, sin): (
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
        GraphTensor<(SequenceLength, Const<ATTENTION_HEAD_DIM>)>,
    ),
) -> GraphTensor<(
    Batch,
    NumAttentionHeads,
    SequenceLength,
    Const<ATTENTION_HEAD_DIM>,
)> {
    let input_half: GraphTensor<(
        Batch,
        NumAttentionHeads,
        SequenceLength,
        Const<ATTENTION_HEAD_DIM>,
    )> = rotate_half::<_, _, _, ATTENTION_HEAD_DIM>(input);

    (input * cos.expand()) + (input_half * sin.expand())
}
