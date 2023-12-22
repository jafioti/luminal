use half::{bf16, f16};
use itertools::Itertools;
use luminal::{
    nn::{linear::Linear, norm::RMSNorm},
    prelude::*,
};
use memmap2::{Mmap, MmapOptions};
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
};
use safetensors::{tensor::TensorView, SafeTensors};
use std::{
    collections::HashMap,
    fs::File,
    ops::{Add, Div},
};
use yoke::{Yoke, Yokeable};

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const NUM_LAYERS: usize = 32;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const ATTENTION_PROJECTION_DIM: usize = 1024;
pub const MLP_PROJECTION_DIM: usize = 14336;

pub const ATTENTION_HEAD_DIM: usize = HIDDEN_DIM / NUM_ATTENTION_HEADS;
pub const ATTENTION_HEAD_DIM_OVER_2: usize = ATTENTION_HEAD_DIM / 2;

// Helper to deserialize safetensors stored in bf16
pub fn convert_vector_bf16_f32(tensor_view: &TensorView<'_>) -> Vec<f32> {
    // Get the data
    let data = tensor_view.data();

    // Create a mutable vector to store the final output
    let mut output: Vec<f32> = Vec::with_capacity(data.len() / 2);

    // Iterate over the raw buffer in chunks of 2 bytes
    for chunk in data.chunks_exact(2) {
        let value = bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32();
        output.push(value);
    }

    output
}

// Rotary Embedding helpers
fn rotate_half<Batch: Dimension, NumHeads: Dimension, Seq: Dimension>(
    x: GraphTensor<(Batch, NumHeads, Seq, Const<ATTENTION_HEAD_DIM>)>,
) -> GraphTensor<(Batch, NumHeads, Seq, Const<ATTENTION_HEAD_DIM>)> {
    let x1 = x
        .slice((.., .., .., ..ATTENTION_HEAD_DIM_OVER_2))
        .contiguous();
    let x2 = x
        .slice((.., .., .., ATTENTION_HEAD_DIM_OVER_2..))
        .contiguous();
    (-x2).concat_along::<(Batch, NumHeads, Seq, Const<ATTENTION_HEAD_DIM>), Axis<3>, _>(x1)
}

// Create the self-Attention layer
pub struct Attention {
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub k_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub rotary_embedding_frequencies: GraphTensor<R1<ATTENTION_HEAD_DIM_OVER_2>>,
}

/*
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

*/

impl Attention {
    // Helper to get a graph
    fn graph(&self) -> &mut Graph {
        self.q_proj.graph()
    }

    // Method to apply rotary embedding
    fn apply_rotary_embeddings<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        xq: GraphTensor<(
            Batch,
            Const<NUM_ATTENTION_HEADS>,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM>,
        )>,
        xk: GraphTensor<(
            Batch,
            Const<NUM_ATTENTION_HEADS>,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM>,
        )>,
    ) -> (
        GraphTensor<(
            Batch,
            Const<NUM_ATTENTION_HEADS>,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM>,
        )>,
        GraphTensor<(
            Batch,
            Const<NUM_ATTENTION_HEADS>,
            SequenceLength,
            Const<ATTENTION_HEAD_DIM>,
        )>,
    ) {
        let graph = self.graph();
        let t = graph.arange::<SequenceLength>();
        let frequencies = t.expand::<(SequenceLength, Const<1>), _>().matmul(
            self.rotary_embedding_frequencies
                .expand::<R2<1, ATTENTION_HEAD_DIM_OVER_2>, _>(),
        );
        let embeddings = frequencies
            .concat_along::<(SequenceLength, Const<ATTENTION_HEAD_DIM>), Axis<1>, _>(frequencies);

        let (sin, cos) = (embeddings.sin(), embeddings.cos());

        let q_embed = (rotate_half(xq) * sin.expand()) + (xq * cos.expand());
        let k_embed = (rotate_half(xk) * sin.expand()) + (xk * cos.expand());

        (q_embed, k_embed)
    }

    // Forward method
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        let mut xq = x
            .matmul(self.q_proj.permute())
            .dyn_reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>(vec![
                Batch::const_size(),
                SequenceLength::const_size(),
                NUM_ATTENTION_HEADS.into(),
                ATTENTION_HEAD_DIM.into(),
            ])
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let mut xk = x
            .matmul(self.k_proj.permute())
            .dyn_reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>(vec![
                Batch::const_size(),
                SequenceLength::const_size(),
                NUM_ATTENTION_HEADS.into(),
                ATTENTION_HEAD_DIM.into(),
            ])
            .permute::<_, Axes4<0, 2, 1, 3>>();
        let xv = x
            .matmul(self.v_proj.permute())
            .dyn_reshape::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            )>(vec![
                Batch::const_size(),
                SequenceLength::const_size(),
                NUM_ATTENTION_HEADS.into(),
                ATTENTION_HEAD_DIM.into(),
            ])
            .permute::<_, Axes4<0, 2, 1, 3>>();

        // We apply rotary embeddings
        (xq, xk) = self.apply_rotary_embeddings(xq, xk);

        // Attention mask
        let attention_mask =
            self.graph().triu::<SequenceLength, SequenceLength>(1) * f16::MIN.to_f32();

        // Finally we compute the outputs (attention calculation)
        let xo = xq
            .matmul(xk.permute())
            .div((ATTENTION_HEAD_DIM as f64).sqrt() as f32)
            .add(attention_mask.expand())
            .softmax::<3>()
            .matmul(xv)
            .permute::<(
                Batch,
                SequenceLength,
                Const<NUM_ATTENTION_HEADS>,
                Const<ATTENTION_HEAD_DIM>,
            ), _>()
            .dyn_reshape::<(Batch, SequenceLength, Const<HIDDEN_DIM>)>(vec![
                Batch::const_size(),
                SequenceLength::const_size(),
                HIDDEN_DIM.into(),
            ])
            .matmul(self.o_proj.permute());

        return xo;
    }
}

// Create the FeedForward Layer
pub struct FeedForward {
    pub gate_proj: GraphTensor<R2<MLP_PROJECTION_DIM, HIDDEN_DIM>>,
    pub down_proj: GraphTensor<R2<HIDDEN_DIM, MLP_PROJECTION_DIM>>,
    pub up_proj: GraphTensor<R2<MLP_PROJECTION_DIM, HIDDEN_DIM>>,
}

impl FeedForward {
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        (x.matmul(self.gate_proj.permute()).swish() * x.matmul(self.up_proj.permute()))
            .matmul(self.down_proj.permute())
    }
}

// Create the Transformer Block
pub struct TransformerBlock {
    pub attention: Attention,
    pub attention_norm: RMSNorm<HIDDEN_DIM>,
    pub feed_forward: FeedForward,
    pub feed_forward_norm: RMSNorm<HIDDEN_DIM>,
}

impl TransformerBlock {
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        let r = self.attention.forward(self.attention_norm.forward(x));
        let h = x + r;
        let r = self.feed_forward.forward(self.feed_forward_norm.forward(h));
        h + r
    }
}

pub struct Mistral {
    // Graph
    pub graph: Box<Graph>,

    // Tokenizer
    pub tokenizer: SentencePieceBpeTokenizer,

    // Embedding
    pub embedding: GraphTensor<R2<VOCAB_SIZE, HIDDEN_DIM>>,

    // Transformer Layers
    pub transformer_layers: Vec<TransformerBlock>,

    // Final Norm layer
    pub norm: RMSNorm<HIDDEN_DIM>,

    // LM Head Layer
    pub lm_head: Linear<HIDDEN_DIM, VOCAB_SIZE>,
}

#[derive(Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

fn get_tensor<'a>(
    weight_file_mapper: &'a HashMap<String, String>,
    file_tensor_mapper: &'a HashMap<String, Yoke<SafeTensors_<'static>, Mmap>>,
    weight_name: &'a str,
) -> Result<TensorView<'a>, String> {
    let file_path = weight_file_mapper
        .get(weight_name)
        .ok_or("Weight not found".to_string())?;

    let tensors = file_tensor_mapper
        .get(file_path)
        .ok_or("Tensors not found".to_string())?;

    let tensor = tensors
        .get()
        .0
        .tensor(weight_name)
        .map_err(|e| e.to_string())?;

    Ok(tensor)
}

impl Mistral {
    // Forward pass
    pub fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &mut self,
        input: GraphTensor<(Batch, SequenceLength)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<VOCAB_SIZE>)> {
        // First compute the embeddings
        let mut hidden_states = self.embedding.gather(input);

        // Loop over transformer layers
        for transformer_layer in &self.transformer_layers {
            hidden_states = transformer_layer.forward(hidden_states);
        }

        // Normalize
        hidden_states = self.norm.forward(hidden_states);

        // Compute output
        self.lm_head.forward(hidden_states)
    }

    // Initializer
    pub fn new(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        // Load the tokenizer
        let tokenizer = SentencePieceBpeTokenizer::from_file(tokenizer_path, false)?;

        // Create the graph
        let mut graph = Box::new(Graph::new());

        // Create the embedding
        let embedding = graph.tensor();

        // Create the transformer layers
        let transformer_layers = (0..NUM_LAYERS)
            .map(|i| TransformerBlock {
                attention: Attention {
                    q_proj: graph.tensor(),
                    k_proj: graph.tensor(),
                    v_proj: graph.tensor(),
                    o_proj: graph.tensor(),
                    rotary_embedding_frequencies: graph.tensor(),
                },
                attention_norm: RMSNorm::initialize(graph.as_mut()),
                feed_forward: FeedForward {
                    gate_proj: graph.tensor(),
                    down_proj: graph.tensor(),
                    up_proj: graph.tensor(),
                },
                feed_forward_norm: RMSNorm::initialize(graph.as_mut()),
            })
            .collect_vec();

        // Create the norm
        let norm = RMSNorm::initialize(graph.as_mut());

        // Create the lm head
        let lm_head = Linear::initialize(graph.as_mut());

        Ok(Self {
            tokenizer,
            graph,
            embedding,
            transformer_layers,
            norm,
            lm_head,
        })
    }

    // Method to encode text as vector
    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let mut vector = self
            .tokenizer
            .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
            .token_ids
            .iter()
            .map(|&x| x as f32)
            .collect_vec();

        vector.insert(0, 1.0); // Start token

        vector
    }

    pub unsafe fn load_safe_tensors_from_files(
        &mut self,
        file_paths: Vec<String>,
    ) -> Result<(), String> {
        let mut weight_file_mapper = HashMap::new();
        let mut file_tensor_mapper = HashMap::new();

        // First, we read the tensors from all the files
        for file_path in file_paths.iter() {
            let file = File::open(file_path).map_err(|e| e.to_string())?;
            let buffer = MmapOptions::new().map(&file).map_err(|e| e.to_string())?;
            let tensors =
                Yoke::<SafeTensors_<'static>, Mmap>::try_attach_to_cart(buffer, |data| {
                    let tensors = SafeTensors::deserialize(data).map_err(|e| e.to_string())?;
                    Ok::<_, String>(SafeTensors_(tensors))
                })?;

            for name in tensors.get().0.names() {
                weight_file_mapper.insert(name.to_string(), file_path.to_string());
            }

            file_tensor_mapper.insert(file_path.to_string(), tensors);
        }

        // Then we populate the layers

        // Layer: Embeddings
        let embeddings_safe_tensor = &get_tensor(
            &weight_file_mapper,
            &file_tensor_mapper,
            "model.embed_tokens.weight",
        )?;
        let embeddings = convert_vector_bf16_f32(embeddings_safe_tensor);
        self.embedding.set(embeddings);

        // Layers: Transformer Layers
        for layer_index in 0..self.transformer_layers.len() {
            // We populate each layer here
            let weight_prefix = format!("model.layers.{layer_index}");

            // Attention
            let weight_name = format!("{weight_prefix}.self_attn.q_proj.weight");
            let q_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;

            let q_proj = convert_vector_bf16_f32(q_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .q_proj
                .set(q_proj);

            let weight_name = format!("{weight_prefix}.self_attn.k_proj.weight");
            let k_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let k_proj = convert_vector_bf16_f32(k_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .k_proj
                .set(k_proj);

            let weight_name = format!("{weight_prefix}.self_attn.v_proj.weight");
            let v_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let v_proj = convert_vector_bf16_f32(v_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .v_proj
                .set(v_proj);

            let weight_name = format!("{weight_prefix}.self_attn.o_proj.weight");
            let o_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let o_proj = convert_vector_bf16_f32(o_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .attention
                .o_proj
                .set(o_proj);

            // Norms
            let weight_name = format!("{weight_prefix}.input_layernorm.weight");
            let attention_norm_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let attention_norm = convert_vector_bf16_f32(attention_norm_safe_tensor);
            self.transformer_layers[layer_index]
                .attention_norm
                .weight
                .set(attention_norm);

            let weight_name = format!("{weight_prefix}.post_attention_layernorm.weight");
            let feed_forward_norm_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let feed_forward_norm = convert_vector_bf16_f32(feed_forward_norm_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward_norm
                .weight
                .set(feed_forward_norm);

            // Feed Forward
            let weight_name = format!("{weight_prefix}.mlp.gate_proj.weight");
            let gate_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let gate_proj = convert_vector_bf16_f32(gate_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .gate_proj
                .set(gate_proj);

            let weight_name = format!("{weight_prefix}.mlp.down_proj.weight");
            let down_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let down_proj = convert_vector_bf16_f32(down_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .down_proj
                .set(down_proj);

            let weight_name = format!("{weight_prefix}.mlp.up_proj.weight");
            let up_proj_safe_tensor = &get_tensor(
                &weight_file_mapper,
                &file_tensor_mapper,
                weight_name.as_str(),
            )?;
            let up_proj = convert_vector_bf16_f32(up_proj_safe_tensor);
            self.transformer_layers[layer_index]
                .feed_forward
                .up_proj
                .set(up_proj);
        }

        // Layer: Norm
        let norm_safe_tensor = &get_tensor(
            &weight_file_mapper,
            &file_tensor_mapper,
            "model.norm.weight",
        )?;
        let norm = convert_vector_bf16_f32(norm_safe_tensor);
        self.norm.weight.set(norm);

        // Layer: LM Head
        let lm_head_safe_tensor =
            &get_tensor(&weight_file_mapper, &file_tensor_mapper, "lm_head.weight")?;
        let lm_head = convert_vector_bf16_f32(lm_head_safe_tensor);
        self.lm_head.weight.set(lm_head);

        Ok(())
    }
}
