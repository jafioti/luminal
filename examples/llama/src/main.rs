use std::{
    collections::HashMap,
    io::{self, Write},
    time::Instant,
};

use clap::Parser;
use itertools::Itertools;
use luminal_2::{
    codegen::codegen,
    run::{produce_buffer_map, run_graph},
    translate::translate_graph,
};
use luminal_metal::{Device, MTLResourceOptions};
use model::{HEAD_DIM, N_KV_HEADS};
use rand::{rng, Rng};
use rustc_hash::FxHashMap;
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::{
    loader::{load_param_f32, q8_load_new},
    model::{KVCache, HIDDEN_DIM, VOCAB_SIZE},
};
use luminal::prelude::{petgraph::algo::toposort, *};

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "256")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/merge_sort.txt"))]
    prompt: String,
}

fn main() {
    let cli_args = CLIArgs::parse();
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    print!("Defining graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();

    // Set up graph
    let mut cx = Graph::new();
    let mut input = cx.named_tensor("Input", 's');
    let mut cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx.named_tensor("Key Cache", (N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();
    cache_src.set_dyn(vec![], (model::N_KV_HEADS, 0, model::HEAD_DIM));
    let model = model::Llama::new(&mut cx);
    let mut model_weights = params(&model);
    // cx.keep_tensors(&model_weights);
    // let logits = model.forward(input);
    let (logits, mut cache_dest) = model.forward((input, &cache_src));
    let mut logits = logits.retrieve();
    cache_dest.retrieve();
    println!("\t\t - {}ms", now.elapsed().as_millis());

    println!("Compiling graph");
    // Set up model loading
    loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);
    let old_logits = logits;
    let old_cache_src = cache_src.clone();
    let old_cache_dest = cache_dest.clone();
    cx.compile(
        (
            // GenericCompiler::default(),
            (
                luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                // luminal_metal::quantized::MetalQuantizedCompiler::<f16>::new(q_weights),
                luminal_metal::BufferCompilers::default(),
            ),
        ),
        (
            // &mut input,
            &mut logits,
            &mut cache_src,
            &mut cache_dest,
            // &mut model_weights,
        ),
    );
    let mut rng = rng();
    let input_data = vec![0.0, 1.0];
    input.set(input_data.clone());
    cx.set_dyn_dim('s', 2);
    cx.set_dyn_dim('p', 0);
    cx.execute_debug();
    // cx.display();
    println!(
        "1.0 kernels: {}",
        cx.node_weights()
            .filter(|n| {
                let s = format!("{:?}", n);
                !s.contains("Load")
                    && !s.contains("Allocate")
                    && !s.contains("Copy")
                    && !s.contains("Execute")
                    && !s.contains("Constant")
            })
            .count()
    );

    let mut outputs = vec![old_to_new_mapping[&old_logits.id]];
    for (k, v) in &old_cache_dest {
        outputs.push(old_to_new_mapping[&k.id]);
        outputs.push(old_to_new_mapping[&v.id]);
    }
    println!("generatin g");
    let kernels = codegen(
        new_graph,
        outputs,
        luminal_2::GPUArch::Metal(HashMap::default()),
        0,
        &cx.dyn_map,
    )
    .unwrap();
    // luminal_2::utils::display_graph(&kernels, &[]);
    println!("2.0 kernels: {}", kernels.node_count() - 2);
    // println!(
    //     "{:?}",
    //     toposort(&cx.graph, None)
    //         .unwrap()
    //         .into_iter()
    //         .map(|n| format!("{:?}", cx.node_weight(n).unwrap()))
    //         .collect_vec()
    // );
    // luminal_2::utils::display_graph(&kernels, &[]);
    // luminal_2::utils::print_kernels(&kernels);
    println!("input: {:?}", old_to_new_mapping[&input.id]);
    let mut inps = vec![(
        old_to_new_mapping[&input.id],
        Box::new(move || input_data) as Box<dyn FnOnce() -> Vec<f32>>,
    )];
    for (k, v) in &old_cache_src {
        println!(
            "k {:?} v {:?}",
            old_to_new_mapping[&k.id], old_to_new_mapping[&v.id]
        );
        inps.push((old_to_new_mapping[&k.id], Box::new(|| vec![])));
        inps.push((old_to_new_mapping[&v.id], Box::new(|| vec![])));
    }
    for (node, val) in q8_load_new("setup/llama3-8b.gguf", &model) {
        inps.push((old_to_new_mapping[&node], val));
    }
    for (label, val) in accs {
        inps.push((label, Box::new(move || val)));
    }

    // let (buf_sizes, buf_map) = produce_buffer_map(&kernels);
    // println!("bufs: {:?}", buf_sizes);
    let (mut outputs, runtime) = run_graph(inps, &kernels, &cx.dyn_map);
    let logits = logits.data();
    println!("Old Logits: {:?}", &logits[..10]);
    println!("New Logits: {:?}", &outputs[0][..10]);
    for (i, (a, b)) in logits.into_iter().zip(outputs.remove(0)).enumerate() {
        if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
            panic!("Index {i} : {a} != {b}");
        }
    }
    let mut kv_dest = cache_dest
        .iter()
        .rev()
        .flat_map(|n| [n.0.data(), n.1.data()])
        .collect_vec();
    let k = kv_dest.remove(0);
    println!("Old K: {:?}", &k[..10]);
    for i in 0..outputs.len() {
        println!("New K: {:?}", &outputs[i][..10]);
    }
    for (i, (a, b)) in k.into_iter().zip(outputs.remove(0)).enumerate() {
        if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
            panic!("Index {i} : {a} != {b}");
        }
    }
    // let v = cache_dest[0].1.data();
    // println!("Old V: {:?}", &v[..10]);
    // println!("New V: {:?}", &outputs[0][..10]);
    // for (i, (a, b)) in v.into_iter().zip(outputs.remove(0)).enumerate() {
    //     if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
    //         panic!("Index {i} : {a} != {b}");
    //     }
    // }
    println!("SUCCESS!");

    // logits.drop();
    // transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    // println!("\t\t - {}ms", now.elapsed().as_millis());

    // // Now that weights are loaded, delete the loading nodes so they don't run again
    // delete_inputs(&cache_src, &mut cx);
    // delete_inputs(downstream(model_weights, &cx), &mut cx);

    // // Run prompt processing pass
    // let input_ids = tokenizer
    //     .encode(&cli_args.prompt as &str, false)
    //     .unwrap()
    //     .get_ids()
    //     .to_vec();
    // input.set_dyn(
    //     input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
    //     (1, input_ids.len()),
    // );
    // cx.set_dyn_dim('t', input_ids.len());
    // print!("Processing Prompt");
    // io::stdout().flush().unwrap();
    // let now = Instant::now();
    // cx.execute();
    // let elapsed_ms = now.elapsed().as_millis();
    // println!(
    //     "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
    //     1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
    //     input_ids.len()
    // );
    // let mut output_ids = vec![argmax(&logits.data())];
    // logits.drop();

    // // Decode token
    // print!("{}", cli_args.prompt.white().bold());
    // let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    // print!("{initial}",);
    // io::stdout().flush().unwrap();

    // // Swap caches
    // transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

    // // Decode loop
    // let start_decode = std::time::Instant::now();
    // let mut prev_output_len = initial.len();
    // for _ in 0..cli_args.gen_tokens {
    //     input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
    //     cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
    //     cx.execute();

    //     // Sample tokens
    //     let output_id = argmax(&logits.data());
    //     logits.drop();
    //     output_ids.push(output_id);

    //     // Get the current decoded output
    //     let current_output = tokenizer.decode(&output_ids, false).unwrap();

    //     // Print the new substring added to the decoded output
    //     print!("{}", current_output[prev_output_len..].bright_green());
    //     io::stdout().flush().unwrap();

    //     // Update the previous output
    //     prev_output_len = current_output.len();

    //     // Swap caches
    //     transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    // }

    // println!();
    // let avg_token_time =
    //     start_decode.elapsed().as_micros() as f32 / (output_ids.len() - 1) as f32 / 1000.0;
    // println!(
    //     "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
    //     avg_token_time,
    //     1000.0 / avg_token_time
    // );
}

// Currently just an argmax, do actual sampling here
fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
