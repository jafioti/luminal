use std::{
    io::{self, Write},
    marker::PhantomData,
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::model::KVCache;
use luminal::{prelude::*, shape::symbolic::Expression};

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "128")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/asimov.txt"))]
    prompt: String,
}

fn main() {
    let cli_args = CLIArgs::parse();
    let tokenizer = Tokenizer::from_file("setup/mistral_tokenizer.json").unwrap();

    print!("Defining graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();

    // Set up graph
    let mut cx = Graph::new();
    let mut input = cx.named_tensor::<(Const<1>, Dyn<'s'>)>("Input");
    let mut cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..model::NUM_LAYERS)
        .map(|_| (cx.named_tensor("Key Cache"), cx.named_tensor("Value Cache")))
        .collect();
    cache_src.set_dyn(vec![], &[1, model::N_KV_HEADS, 0, model::HEAD_DIM]);
    let model = model::MistralLM::initialize(&mut cx);
    let mut model_weights = downstream(params(&model), &cx);
    cx.keep_tensors(&model_weights);
    let (logits, mut cache_dest) = model.forward((input, &cache_src, PhantomData::<Dyn<'t'>>));
    let mut logits = logits
        .slice((.., (Expression::from('s') - 1).., ..))
        .retrieve();
    cache_dest.keep();

    // Set up model loading
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let quantized_weight_nodes =
        loader::Q8Loader::new("setup/mistral-7b-instruct-v0.2.Q8_0.gguf").load(&model, &mut cx);
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    loader::Q8Loader::new("setup/mistral-7b-instruct-v0.2.Q8_0.gguf").load(&model, &mut cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    print!("Compiling graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(quantized_weight_nodes),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaQuantizedCompiler::<f32>::new(quantized_weight_nodes),
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
    // cx.display();

    // Keep model weights
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
    delete_inputs(&model_weights, &mut cx);
    // Run prompt processing pass
    let mut input_ids = encode(&tokenizer, &cli_args.prompt);
    input_ids.insert(0, 1);
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        &[1, input_ids.len()],
    );
    cx.set_dyn_dim('t', input_ids.len());
    print!("Processing Prompt");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.execute();
    let elapsed_ms = now.elapsed().as_millis();
    println!(
        "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
        1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
        input_ids.len()
    );
    delete_inputs(&cache_src_set, &mut cx);
    let output_id = sample_index(&logits.data());
    logits.drop();
    input_ids.push(output_id);

    let mut output_ids = vec![output_id];

    // Decode token
    print!("{}", cli_args.prompt.white().bold());
    io::stdout().flush().unwrap();

    // Swap caches
    transfer_data_same_graph(&cache_dest_set, &cache_src_set, &mut cx);

    // Decode loop
    let mut token_decode_times = vec![];
    let mut prev_output_len = 0;
    for _ in 0..cli_args.gen_tokens {
        // for _ in 0..1 {
        input.set_dyn(vec![*input_ids.last().unwrap() as f32], &[1, 1]);
        cx.set_dyn_dim('p', input_ids.len() - 1);
        cx.set_dyn_dim('t', input_ids.len());
        let now = Instant::now();
        cx.execute();
        token_decode_times.push(now.elapsed().as_micros());

        // Sample tokens
        let output_id = sample_index(&logits.data());
        logits.drop();
        input_ids.push(output_id);
        output_ids.push(output_id);

        // Get the current decoded output
        let current_output = decode(&tokenizer, &output_ids);

        // Print the new substring added to the decoded output
        let new_substring = &current_output[prev_output_len..];
        print!("{}", new_substring.bright_green());
        io::stdout().flush().unwrap();

        // Update the previous output
        prev_output_len = current_output.len();

        // Swap caches
        transfer_data_same_graph(&cache_dest_set, &cache_src_set, &mut cx);
    }

    println!();
    let avg_token_time = token_decode_times
        .iter()
        .map(|t| *t as f32 / 1000.)
        .sum::<f32>()
        / token_decode_times.len() as f32;
    println!(
        "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
        avg_token_time,
        1000.0 / avg_token_time
    );
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let vector = tokenizer.encode(text, false).unwrap();
    vector.get_ids().to_owned()
}

fn decode(tokenizer: &Tokenizer, token_ids: &[u32]) -> String {
    tokenizer.decode(token_ids, false).unwrap()
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> u32 {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0 as u32
}
