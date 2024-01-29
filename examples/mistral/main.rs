use std::{
    io::{self, Write},
    marker::PhantomData,
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use half::f16;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};
mod loader;
mod model;

use luminal::{prelude::*, shape::symbolic::Expression};

use crate::model::KVCache;

#[cfg(feature = "metal")]
type DeviceCompiler = MetalCompiler<f16>;
#[cfg(feature = "cuda")]
type DeviceCompiler = CudaFp16Compiler;
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
type DeviceCompiler = CPUCompiler;

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "256")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("prompts/asimov.txt"))]
    prompt: String,
}

fn main() {
    let cli_args = CLIArgs::parse();
    let tokenizer = SentencePieceBpeTokenizer::from_file(
        "./examples/mistral/setup/mistral-7b-hf/tokenizer.model",
        false,
    )
    .unwrap();

    print!("Defining Graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    let mut cx = Graph::new();
    let mut input = cx.named_tensor::<(Const<1>, Dyn<'s'>)>("Input");
    let mut cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..model::NUM_LAYERS)
        .map(|_| (cx.named_tensor("Key Cache"), cx.named_tensor("Value Cache")))
        .collect();
    cache_src.set_dyn(vec![], &[1, model::N_KV_HEADS, 0, model::HEAD_DIM]);
    let model = model::MistralLM::initialize(&mut cx);
    let (logits, mut cache_dest) =
        model.forward((input, Some(cache_src.clone()), PhantomData::<Dyn<'t'>>));
    let mut logits = logits
        .slice((.., (Expression::from('s') - 1).., ..))
        .retrieve();
    cache_dest.keep();

    // Set up model loading
    loader::MetalFp16SafetensorsLoader::new(&[
        "./examples/mistral/setup/mistral-7b-hf/converted-model-00001-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/converted-model-00002-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/converted-model-00003-of-00003.safetensors",
    ])
    .load(&model, &mut cx);
    println!("\t - {}ms", now.elapsed().as_millis());

    print!("Compiling Graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.compile(
        <(GenericCompiler, DeviceCompiler)>::default(),
        (&mut input, &mut logits, &mut cache_src, &mut cache_dest),
    );
    let model_weights = downstream(&state_set(&model), &cx);
    cx.keep_tensors(&model_weights);
    let cache_src_set = downstream(&cache_src, &cx);
    let cache_dest_set = cache_dest.to_ids();
    println!("\t - {}ms", now.elapsed().as_millis());

    // Initial forward pass to load weights
    print!("Loading model");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    input.set_dyn(vec![0.], &[1, 1]);
    cx.set_dyn_dim('t', 1);
    cx.execute();
    logits.drop();
    cache_dest.drop();
    println!("\t - {}ms", now.elapsed().as_millis());

    // Now that weights are loaded, delete the loading nodes so they don't run again
    delete_inputs(&model_weights, &mut cx);

    // Run inference first pass
    let mut input_ids = encode(&tokenizer, &cli_args.prompt);
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
        "\t - {elapsed_ms}ms ({:.2} tok/s)",
        1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64)
    );
    delete_inputs(&cache_src_set, &mut cx);
    let output_id = sample_index(&logits.data());
    logits.drop();
    input_ids.push(output_id);

    // Decode token
    print!(
        "{}{}",
        cli_args.prompt.white().bold(),
        decode(&tokenizer, &[output_id]).bright_green()
    );
    io::stdout().flush().unwrap();

    // Swap caches
    transfer_data_same_graph(&cache_dest_set, &cache_src_set, &mut cx);

    // Decode loop
    let mut token_decode_times = vec![];
    for _ in 0..cli_args.gen_tokens {
        input.set_dyn(vec![*input_ids.last().unwrap() as f32], &[1, 1]);
        cx.set_dyn_dim('p', input_ids.len() - 1);
        cx.set_dyn_dim('t', input_ids.len());

        let now = Instant::now();
        cx.execute();
        token_decode_times.push(now.elapsed().as_millis());

        // Sample tokens
        let output_id = sample_index(&logits.data());
        logits.drop();
        input_ids.push(output_id);
        print!("{}", decode(&tokenizer, &[output_id]).bright_green());
        io::stdout().flush().unwrap();

        // Swap caches
        transfer_data_same_graph(&cache_dest_set, &cache_src_set, &mut cx);
    }
    println!(
        "\nAverage token generated in {}ms",
        token_decode_times.iter().sum::<u128>() / token_decode_times.len() as u128
    );
}

fn encode(tokenizer: &SentencePieceBpeTokenizer, text: &str) -> Vec<i64> {
    let mut vector = tokenizer
        .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
        .token_ids;
    vector.insert(0, 1); // Start token
    vector
}

fn decode(tokenizer: &SentencePieceBpeTokenizer, token_ids: &[i64]) -> String {
    tokenizer
        .decode(token_ids, true, false)
        .replace("<0x0A>", "\n")
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> i64 {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0 as i64
}
