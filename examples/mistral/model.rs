use half::{bf16, f16};
use itertools::Itertools;
use luminal::{nn::linear::Linear, prelude::*};
use memmap2::{Mmap, MmapOptions};
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
};
use safetensors::{tensor::TensorView, SafeTensorError, SafeTensors};
use std::{
    fs::File,
    ops::{Add, Div},
};

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const LAYERS: usize = 32;
pub const ATTENTION_HEAD_DIM: usize = HIDDEN_DIM / LAYERS;
pub const NUM_ATTENTION_HEADS: usize = 32;
pub const ATTENTION_PROJECTION_DIM: usize = 1024;
pub const MLP_PROJECTION_DIM: usize = 14336;

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

// Rotary Embedding Layer
pub struct RotaryEmbedding<const HEAD_DIM: usize, const HEAD_DIM_OVER_2: usize> {
    pub inv_freq: GraphTensor<R1<HEAD_DIM_OVER_2>>,
}

// Create the self-Attention layer
pub struct Attention {
    pub q_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub k_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub v_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
    pub o_proj: GraphTensor<R2<HIDDEN_DIM, HIDDEN_DIM>>,
}

impl Attention {
    fn forward<Batch: Dimension, SequenceLength: Dimension>(
        &self,
        x: GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)>,
    ) -> GraphTensor<(Batch, SequenceLength, Const<HIDDEN_DIM>)> {
        let xq = x
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
        let xk = x
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

        // We apply rotary embeddings here

        // Attention mask
        let attention_mask = self
            .q_proj
            .graph()
            .triu::<SequenceLength, SequenceLength>(1)
            * f16::MIN.to_f32();

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

pub struct Mistral {
    // Graph
    pub graph: Box<Graph>,

    // Tokenizer
    pub tokenizer: SentencePieceBpeTokenizer,

    // Embedding
    pub embedding: Linear<VOCAB_SIZE, HIDDEN_DIM>,
    // Decoder Layers
    // pub layers: Vec<DecoderLayer>
}

impl Mistral {
    // Initializer
    pub fn new(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        // Load the tokenizer
        let tokenizer = SentencePieceBpeTokenizer::from_file(tokenizer_path, false)?;

        // Create the graph
        let mut graph = Box::new(Graph::new());

        // Create the embedding
        let embedding = Linear::initialize(graph.as_mut());

        Ok(Self {
            tokenizer,
            graph,
            embedding,
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

    // Method to load weights from file
    pub fn load_safe_tensors_from_file(&mut self, filename: &str) -> Result<(), String> {
        let file = File::open(filename).map_err(|e| e.to_string())?;

        let buffer = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };

        let tensors = SafeTensors::deserialize(&buffer).map_err(|e| e.to_string())?;

        let _ = self
            .load_safe_tensors(&tensors)
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    // Method to load weights
    pub fn load_safe_tensors(&mut self, tensors: &SafeTensors<'_>) -> Result<(), SafeTensorError> {
        // Pull in the embeddings
        let embeddings_safe_tensor = tensors.tensor("model.embed_tokens.weight")?;

        // Convert to f32
        let embeddings = convert_vector_bf16_f32(&embeddings_safe_tensor);

        // Apply to embeddings layer
        self.embedding.weight.set(embeddings);

        Ok(())
    }
}
