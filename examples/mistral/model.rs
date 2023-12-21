use itertools::Itertools;
use luminal::prelude::*;
use rust_tokenizers::{
    error::TokenizerError,
    tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy},
};

// Mistral 7B Config
pub const VOCAB_SIZE: usize = 32000;
pub const HIDDEN_DIM: usize = 4096;
pub const LAYERS: usize = 32;
pub const ATTENTION_PROJECTION_DIM: usize = 1024;
pub const MLP_PROJECTION_DIM: usize = 14336;

pub struct Mistral {
    pub tokenizer: SentencePieceBpeTokenizer,
    // pub next_token_graph: *mut Graph,
    // pub prompt_processing_graph: *mut Graph,
}

impl Mistral {
    pub fn new(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        let tokenizer = SentencePieceBpeTokenizer::from_file(tokenizer_path, false)?;

        Ok(Self {
            tokenizer: tokenizer,
        })
    }

    // Method to encode text as vector
    pub fn encode(self, text: &str) -> Vec<f32> {
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
}
