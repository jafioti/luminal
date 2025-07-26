use std::collections::HashMap;

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use luminal_2::{
    codegen::codegen,
    run::run_graph,
    translate::{translate_graph, InitData},
};
use luminal_metal::{Buffer, Device, MTLResourceOptions};
use model::{HEAD_DIM, N_KV_HEADS};
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::{loader::load_new, model::KVCache};
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
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);

    // Codegen
    let mut outputs = vec![old_to_new_mapping[&logits.id]];
    for (k, v) in &cache_dest {
        outputs.push(old_to_new_mapping[&k.id]);
        outputs.push(old_to_new_mapping[&v.id]);
    }
    let kernels = codegen(
        new_graph,
        outputs,
        luminal_2::GPUArch::Metal(HashMap::default()),
        0,
        &cx.dyn_map,
    )
    .unwrap();
    // luminal_2::utils::display_graph(&kernels, &[]);

    // Set up inputs
    let input_ids_clone = input_ids.clone();
    let mut inps = vec![(
        old_to_new_mapping[&input.id],
        Box::new(move || {
            Device::system_default().unwrap().new_buffer_with_data(
                input_ids_clone.as_ptr() as *const _,
                (input_ids_clone.len() * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }) as Box<dyn FnOnce() -> Buffer>,
    )];
    for (k, v) in &cache_src {
        inps.push((
            old_to_new_mapping[&k.id],
            Box::new(|| {
                Device::system_default()
                    .unwrap()
                    .new_buffer(0, MTLResourceOptions::StorageModeShared)
            }),
        ));
        inps.push((
            old_to_new_mapping[&v.id],
            Box::new(|| {
                Device::system_default()
                    .unwrap()
                    .new_buffer(0, MTLResourceOptions::StorageModeShared)
            }),
        ));
    }
    for (node, val) in load_new("setup/llama3-8b.gguf", &model) {
        inps.push((old_to_new_mapping[&node], val));
    }
    for (label, val) in &accs {
        match val {
            InitData::Expr(e) => {
                let val = e.exec(&cx.dyn_map).unwrap();
                inps.push((
                    *label,
                    Box::new(move || {
                        let v = vec![val as f32];
                        Device::system_default().unwrap().new_buffer_with_data(
                            v.as_ptr() as *const _,
                            size_of::<f32>() as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    }),
                ));
            }
            InitData::Data(d) => {
                let d = d.clone();
                inps.push((
                    *label,
                    Box::new(move || {
                        Device::system_default().unwrap().new_buffer_with_data(
                            d.as_ptr() as *const _,
                            (d.len() * size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    }),
                ))
            }
        }
    }

    // Run prompt processing pass
    let (outputs, _) = run_graph(inps, &kernels, &cx.dyn_map);

    // Process outputs
    let mut output_id = argmax(&outputs[0]);
    input_ids.push(output_id as f32);
    let mut kv_out = outputs
        .into_iter()
        .skip(1)
        .chunks(2)
        .into_iter()
        .map(|mut i| (i.next().unwrap(), i.next().unwrap()))
        .collect_vec();

    // Decode token
    let mut completion = tokenizer.decode(&[output_id], false).unwrap();
    println!(
        "{}{}",
        cli_args.prompt.white().bold(),
        completion.bright_green()
    );

    // Decode loop
    for _ in 0..10 {
        cx.set_dyn_dim('s', 1);
        cx.set_dyn_dim('p', input_ids.len() - 1);

        // Set up inputs
        let output_ids_clone = vec![output_id as f32];
        let mut inps = vec![(
            old_to_new_mapping[&input.id],
            Box::new(move || {
                Device::system_default().unwrap().new_buffer_with_data(
                    output_ids_clone.as_ptr() as *const _,
                    (output_ids_clone.len() * size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }) as Box<dyn FnOnce() -> Buffer>,
        )];
        for ((k, v), (k_data, v_data)) in cache_src.iter().zip(kv_out) {
            inps.push((
                old_to_new_mapping[&k.id],
                Box::new(move || {
                    Device::system_default().unwrap().new_buffer_with_data(
                        k_data.as_ptr() as *const _,
                        (k_data.len() * size_of::<f32>()) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                }),
            ));
            inps.push((
                old_to_new_mapping[&v.id],
                Box::new(move || {
                    Device::system_default().unwrap().new_buffer_with_data(
                        v_data.as_ptr() as *const _,
                        (v_data.len() * size_of::<f32>()) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                }),
            ));
        }
        for (node, val) in load_new("setup/llama3-8b.gguf", &model) {
            inps.push((old_to_new_mapping[&node], val));
        }
        for (label, val) in &accs {
            match val {
                InitData::Expr(e) => {
                    let val = e.exec(&cx.dyn_map).unwrap();
                    inps.push((
                        *label,
                        Box::new(move || {
                            let v = vec![val as f32];
                            Device::system_default().unwrap().new_buffer_with_data(
                                v.as_ptr() as *const _,
                                size_of::<f32>() as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        }),
                    ));
                }
                InitData::Data(d) => {
                    let d = d.clone();
                    inps.push((
                        *label,
                        Box::new(move || {
                            Device::system_default().unwrap().new_buffer_with_data(
                                d.as_ptr() as *const _,
                                (d.len() * size_of::<f32>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        }),
                    ))
                }
            }
        }

        // Run
        let (outputs, _) = run_graph(inps, &kernels, &cx.dyn_map);

        // Get outputs
        output_id = argmax(&outputs[0]);
        input_ids.push(output_id as f32);
        kv_out = outputs
            .into_iter()
            .skip(1)
            .chunks(2)
            .into_iter()
            .map(|mut i| (i.next().unwrap(), i.next().unwrap()))
            .collect_vec();

        // Decode token
        completion.push_str(&tokenizer.decode(&[output_id], false).unwrap());
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
