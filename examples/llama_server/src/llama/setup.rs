use std::{
    io::{self, Write},
    marker::PhantomData,
    time::Instant,
};

use luminal::prelude::*;
use tokenizers::Tokenizer;

use crate::llama::{
    loader,
    model::{KVCache, MistralLM, HEAD_DIM, NUM_LAYERS, N_KV_HEADS},
};

/// Define the model
pub struct Model {
    pub graph: Box<Graph>,
    pub input: GraphTensor<(Const<1>, Dyn<'s'>)>,
    pub tokenizer: Tokenizer,
}

// Load the model
impl Model {
    pub fn setup() -> Self {
        let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

        print!("Defining graph");
        let now = Instant::now();

        // Set up graph
        let mut cx = Box::new(Graph::new());

        let mut input = cx.named_tensor::<(Const<1>, Dyn<'s'>)>("Input");
        let mut cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..NUM_LAYERS)
            .map(|_| (cx.named_tensor("Key Cache"), cx.named_tensor("Value Cache")))
            .collect();
        cache_src.set_dyn(vec![], &[1, N_KV_HEADS, 0, HEAD_DIM]);
        let model = MistralLM::initialize(&mut cx);
        let mut model_weights = params(&model);
        cx.keep_tensors(&model_weights);
        let (logits, mut cache_dest) = model.forward((input, &cache_src, PhantomData::<Dyn<'t'>>));
        let mut logits = logits
            .slice((.., (Expression::from('s') - 1).., ..))
            .retrieve();
        cache_dest.keep();

        // Set up model loading
        #[cfg(any(feature = "metal", feature = "cuda"))]
        let q_weights = loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
        println!("\t\t - {}ms", now.elapsed().as_millis());

        print!("Compiling graph");
        io::stdout().flush().unwrap();
        let now = Instant::now();
        cx.compile(
            (
                GenericCompiler::default(),
                #[cfg(feature = "metal")]
                luminal_metal::quantized::MetalQuantizedCompiler::<f16>::new(q_weights),
                #[cfg(feature = "cuda")]
                luminal_cuda::CudaQuantizedCompiler::<f16>::new(q_weights),
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
        let cache_src_set = downstream(&cache_src, &cx);
        let cache_dest_set = cache_dest.to_ids();
        println!("\t\t - {}ms", now.elapsed().as_millis());

        // Initial forward pass to load weights
        print!("Loading model");
        io::stdout().flush().unwrap();
        let now = Instant::now();
        input.set_dyn(vec![1.], &[1, 1]);
        cx.set_dyn_dim('t', 1);
        cx.execute();
        logits.drop();
        cache_dest.drop();
        println!("\t\t - {}ms", now.elapsed().as_millis());

        // Now that weights are loaded, delete the loading nodes so they don't run again
        delete_inputs(&downstream(model_weights, &cx), &mut cx);

        Model {
            graph: cx,
            input,
            tokenizer,
        }
    }
}
