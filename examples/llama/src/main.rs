use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use luminal_2::{
    codegen::codegen,
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph, InitData},
};
use model::{HEAD_DIM, N_KV_HEADS};
use rustc_hash::FxHashMap;
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::{loader::*, model::*};
use luminal::prelude::*;

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
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    panic!("Either metal or cuda feature must be used for this example!");

    let cli_args = CLIArgs::parse();
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();
    let mut input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec()
        .into_iter()
        .map(|i| i as f32)
        .collect_vec();

    // Set up 1.0 graph
    let mut cx = Graph::new();
    let input = cx.named_tensor("Input", (1, 's'));
    let cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();
    let model = model::Llama::new(&mut cx);
    let (logits, cache_dest) = model.forward((input, &cache_src));
    let logits = logits
        .slice((.., Expression::from('s') - 1..))
        .contiguous()
        .retrieve();
    cx.set_dyn_dim('s', input_ids.len());
    cx.set_dyn_dim('p', 0);

    // Convert to 2.0 graph
    let now = std::time::Instant::now();
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);
    println!("Translate: {}ms", now.elapsed().as_millis());

    // Codegen
    let mut outputs = vec![old_to_new_mapping[&logits.id]];
    for (k, v) in &cache_dest {
        outputs.push(old_to_new_mapping[&k.id]);
        outputs.push(old_to_new_mapping[&v.id]);
    }
    let now = std::time::Instant::now();
    let (kernels, gmem_mapping) = codegen(
        new_graph.clone(),
        outputs,
        luminal_2::GPUArch::CUDA,
        0,
        &cx.dyn_map,
    )
    .unwrap();
    println!("Codegen: {}ms", now.elapsed().as_millis());
    let now = std::time::Instant::now();
    let compiled_kernels = compile_kernels(&kernels);
    let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
    println!("Compile: {}ms", now.elapsed().as_millis());

    // Set up inputs
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut inps = FxHashMap::default();
    inps.insert(
        gmem_mapping[&old_to_new_mapping[&input.id]],
        (copy_cuda_buffer(&input_ids, &stream), true),
    );
    for (k, v) in &cache_src {
        inps.insert(
            gmem_mapping[&old_to_new_mapping[&k.id]],
            (stream.alloc_zeros(1).unwrap(), true),
        );
        inps.insert(
            gmem_mapping[&old_to_new_mapping[&v.id]],
            (stream.alloc_zeros(1).unwrap(), true),
        );
    }
    let now = std::time::Instant::now();
    for (node, val) in load("setup/llama3-8b.gguf", &model) {
        inps.insert(gmem_mapping[&old_to_new_mapping[&node]], (val, false));
    }
    println!("Load: {}ms", now.elapsed().as_millis());
    for (label, val) in &accs {
        match val {
            InitData::Expr(e) => {
                let val = e.exec(&cx.dyn_map).unwrap();
                inps.insert(
                    gmem_mapping[label],
                    (copy_cuda_buffer(&vec![val as f32], &stream), true),
                );
            }
            InitData::Data(d) => {
                inps.insert(gmem_mapping[label], (copy_cuda_buffer(&d, &stream), false));
            }
        }
    }

    // Run prompt processing pass
    let (mut outputs, _) = run_graph(
        &mut inps,
        &kernels,
        &cx.dyn_map,
        &compiled_kernels,
        &int_buffers,
        &int_buffer_map,
    );

    // Process outputs
    let mut logits = stream.memcpy_dtov(&outputs[0]).unwrap();
    let mut output_id = argmax(&logits);
    input_ids.push(output_id as f32);
    // let mut kv_out = outputs
    //     .into_iter()
    //     .skip(1)
    //     .chunks(2)
    //     .into_iter()
    //     .map(|mut i| (i.next().unwrap(), i.next().unwrap()))
    //     .collect_vec();

    // Decode token
    let mut completion = tokenizer.decode(&[output_id], false).unwrap();
    println!(
        "{}{}",
        cli_args.prompt.white().bold(),
        completion.bright_green()
    );

    // Decode loop
    for _ in 0..100 {
        let n = std::time::Instant::now();
        cx.set_dyn_dim('s', 1);
        cx.set_dyn_dim('p', input_ids.len() - 1);

        // Set up inputs
        inps.insert(gmem_mapping[&old_to_new_mapping[&input.id]], {
            (copy_cuda_buffer(&vec![output_id as f32], &stream), true)
        });
        for ((k, v), mut kv_data) in cache_src
            .iter()
            .zip(outputs.into_iter().skip(1).chunks(2).into_iter())
        {
            inps.insert(
                gmem_mapping[&old_to_new_mapping[&k.id]],
                (kv_data.next().unwrap(), true),
            );
            inps.insert(
                gmem_mapping[&old_to_new_mapping[&v.id]],
                (kv_data.next().unwrap(), true),
            );
        }
        for (label, val) in &accs {
            match val {
                InitData::Expr(e) => {
                    let val = e.exec(&cx.dyn_map).unwrap();
                    inps.insert(
                        gmem_mapping[label],
                        (copy_cuda_buffer(&vec![val as f32], &stream), true),
                    );
                }
                InitData::Data(_) => {} // Wasn't deleted, don't need to make new buffer
            }
        }
        println!("init {}micros", n.elapsed().as_micros());

        // Run
        (outputs, _) = run_graph(
            &mut inps,
            &kernels,
            &cx.dyn_map,
            &compiled_kernels,
            &int_buffers,
            &int_buffer_map,
        );
        println!("run {}micros", n.elapsed().as_micros());

        // Get outputs
        stream.memcpy_dtoh(&outputs[0], &mut logits).unwrap();
        output_id = argmax(&logits);
        input_ids.push(output_id as f32);

        // Decode token
        completion.push_str(&tokenizer.decode(&[output_id], false).unwrap());
        println!("token {}micros", n.elapsed().as_micros());
        println!(
            "{}{}",
            cli_args.prompt.white().bold(),
            completion.bright_green()
        );
    }
}

// Currently just an argmax, do actual sampling here
fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
