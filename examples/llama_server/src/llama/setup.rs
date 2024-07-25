use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

use itertools::Itertools;
use luminal::prelude::*;
use tokenizers::Tokenizer;

use crate::llama::{
    loader,
    model::{KVCache, Llama, HEAD_DIM, NUM_LAYERS, N_KV_HEADS},
};

/// Define the model
pub struct Model {
    pub graph: Box<Graph>,
    pub input: GraphTensor,
    kv_cache_src_set: Vec<NodeIndex>,
    kv_cache_dest_set: Vec<NodeIndex>,
    logits: GraphTensor,
    pub tokenizer: Tokenizer,
    pub last_generated_token: Option<u32>,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

const TOKENIZER_PATH: &str = "./setup/tokenizer.json";
const MODEL_PATH: &str = "./setup/llama3-8b.gguf";

// Load the model
impl Model {
    pub fn setup() -> Self {
        if Path::new(TOKENIZER_PATH).exists() && Path::new(MODEL_PATH).exists() {
            println!("Tokenizer and Model Exists");
        } else {
            panic!("Model does not exist");
        }

        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();

        print!("Defining graph");
        let now = Instant::now();

        // Set up graph
        let mut cx = Box::new(Graph::new());

        let mut input = cx.named_tensor("Input", (1, 's'));
        let mut cache_src: Vec<KVCache> = (0..NUM_LAYERS)
            .map(|_| {
                (
                    cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                    cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                )
            })
            .collect();
        cache_src.set_dyn(vec![], (1, N_KV_HEADS, 0, HEAD_DIM));
        let model = Llama::new(&mut cx);
        let mut model_weights = params(&model);
        cx.keep_tensors(&model_weights);
        let (logits, mut cache_dest) = model.forward((input, &cache_src));
        let mut logits = logits
            .slice((.., (Expression::from('s') - 1).., ..))
            .retrieve();
        cache_dest.keep();

        // Set up model loading
        #[cfg(any(feature = "metal", feature = "cuda"))]
        let q_weights = loader::q8_load(MODEL_PATH, &model, &mut cx);
        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        loader::q8_load(MODEL_PATH, &model, &mut cx);
        println!("\t\t - {}ms", now.elapsed().as_millis());

        print!("Compiling graph");
        io::stdout().flush().unwrap();
        let now = Instant::now();
        cx.compile(
            (
                GenericCompiler::default(),
                #[cfg(feature = "metal")]
                (
                    luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                    luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
                    luminal_metal::BufferCompilers::default(),
                ),
                #[cfg(feature = "cuda")]
                (
                    luminal_cuda::CudaCompiler::<f16>::default(),
                    luminal_cuda::CudaQuantizedCompiler::<f16>::new(q_weights),
                ),
                #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
                luminal_cpu::CPUCompiler::default(),
            ),
            (
                &mut input,
                &mut logits,
                &mut cache_src,
                &mut cache_dest,
                &mut model_weights,
            ),
        );
        let cache_src = downstream(&cache_src, &cx);
        let cache_dest = cache_dest.to_ids();
        println!("\t\t - {}ms", now.elapsed().as_millis());

        // Initial forward pass to load weights
        print!("Loading model");
        io::stdout().flush().unwrap();
        let now = Instant::now();
        input.set_dyn(vec![1.], (1, 1));
        cx.set_dyn_dim('t', 1);
        cx.execute();
        logits.drop();
        cx.drop_tensors(&cache_dest);
        println!("\t\t - {}ms", now.elapsed().as_millis());

        // Now that weights are loaded, delete the loading nodes so they don't run again
        delete_inputs(downstream(model_weights, &cx), &mut cx);

        Model {
            input,
            tokenizer,
            kv_cache_src_set: downstream(cache_src, &cx),
            kv_cache_dest_set: cache_dest.to_ids(),
            graph: cx,
            logits,
            last_generated_token: None,
        }
    }

    /// Generate new tokens given some input
    pub fn generate(&mut self, prompt: &str, mut continue_callback: impl FnMut(u32) -> bool) {
        let input_tokens = self.tokenizer.encode(prompt, false).unwrap();
        let input_tokens = input_tokens.get_ids();

        self.generate_internal(input_tokens, |dist| {
            let output_id = argmax(dist);
            (output_id, continue_callback(output_id))
        })
    }

    fn generate_internal(
        &mut self,
        prompt: &[u32],
        mut callback: impl FnMut(&[f32]) -> (u32, bool),
    ) {
        const EOS_TOKEN: u32 = 128009; // From the llama3 vocab

        let mut input_ids = prompt.to_vec();

        // Set the dyn dims
        let mut seq_len = input_ids.len();
        let mut p = 0;
        if self.graph.dyn_map[&'p'] != 0 {
            input_ids.insert(0, self.last_generated_token.unwrap());
            p = self.graph.dyn_map[&'t'];
            seq_len += p + 1;
        }

        self.graph.set_dyn_dim('p', p);
        self.graph.set_dyn_dim('t', seq_len);
        self.input.set_dyn(
            input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
            (1, input_ids.len()),
        );

        // First token output (from prompt processing)
        self.graph.execute();

        // Get the output token
        let (mut output_id, mut cont) = callback(&self.logits.data());
        self.logits.drop();
        seq_len += 1;
        self.last_generated_token = Some(output_id);

        // Swap cache
        transfer_data_same_graph(
            &self.kv_cache_dest_set,
            &self.kv_cache_src_set,
            &mut self.graph,
        );

        // Decode loop (next token)
        while output_id != EOS_TOKEN && cont {
            // Set the data
            self.graph.set_dyn_dim('p', seq_len - 1);
            self.graph.set_dyn_dim('t', seq_len);
            self.input.set_dyn(vec![output_id as f32], (1, 1));

            // Execute the graph
            self.graph.execute();

            // Get the output token
            (output_id, cont) = callback(&self.logits.data());
            seq_len += 1;
            self.logits.drop();
            self.last_generated_token = Some(output_id);

            // Swap cache
            transfer_data_same_graph(
                &self.kv_cache_dest_set,
                &self.kv_cache_src_set,
                &mut self.graph,
            );
        }
    }

    pub fn clear_cache(&mut self) {
        self.last_generated_token = None;
        self.graph.set_dyn_dim('p', 0);
    }
}

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
