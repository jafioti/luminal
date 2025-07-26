// use std::{
//     collections::HashMap,
//     io::{self, Write},
//     time::Instant,
// };

// use clap::Parser;
// use colored::Colorize;
// use itertools::Itertools;
// use luminal_2::{
//     codegen::codegen,
//     run::{produce_buffer_map, run_graph},
//     translate::{translate_graph, InitData},
// };
// use luminal_metal::{Device, MTLResourceOptions};
// use model::{HEAD_DIM, N_KV_HEADS};
// use rand::{rng, Rng};
// use rustc_hash::FxHashMap;
// use tokenizers::Tokenizer;

// mod gguf;
// mod loader;
// mod model;

// use crate::model::{KVCache, HIDDEN_DIM, VOCAB_SIZE};
// use luminal::prelude::{petgraph::algo::toposort, *};

// // Command args parser
// #[derive(Debug, Parser)]
// #[command(author, version, about, long_about = None)]
// pub struct CLIArgs {
//     /// Number of tokens to generate
//     #[clap(short = 't', long = "gen_tokens", default_value = "256")]
//     gen_tokens: i32,

//     /// Prompt for the model
//     #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/merge_sort.txt"))]
//     prompt: String,
// }

// fn main() {
//     let cli_args = CLIArgs::parse();
//     let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

//     print!("Defining graph");
//     io::stdout().flush().unwrap();
//     let now = Instant::now();

//     let input_ids = tokenizer
//         .encode(&cli_args.prompt as &str, false)
//         .unwrap()
//         .get_ids()
//         .to_vec();
//     // Set up graph
//     let mut cx = Graph::new();
//     let mut input = cx.named_tensor("Input", (1, 's'));
//     let mut rng = rng();
//     let mut cache_src_data: Vec<(Vec<f32>, Vec<f32>)> = (0..model::NUM_LAYERS)
//         .map(|_| {
//             (
//                 (0..(N_KV_HEADS * 0 * HEAD_DIM))
//                     .map(|_| rng.random())
//                     .collect_vec(),
//                 (0..(N_KV_HEADS * 0 * HEAD_DIM))
//                     .map(|_| rng.random())
//                     .collect_vec(),
//             )
//         })
//         .collect_vec();
//     let mut cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
//         .map(|_| {
//             (
//                 cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
//                 cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
//             )
//         })
//         .collect();
//     cache_src.set_dyn(vec![], (N_KV_HEADS, 0, HEAD_DIM));
//     let model = model::Llama::new(&mut cx);
//     let mut model_weights = params(&model);
//     cx.keep_tensors(&model_weights);
//     // let logits = model.forward(input);
//     let (logits, mut cache_dest) = model.forward((input, &cache_src));
//     let mut logits = logits
//         .slice(Expression::from('s') - 1..)
//         .contiguous()
//         .retrieve();
//     cache_dest.keep();
//     println!("\t\t - {}ms", now.elapsed().as_millis());

//     println!("Compiling graph");
//     // Set up model loading
//     let q_weights = loader::q8_load("setup/llama3-8b.gguf", &model, &mut cx);
//     // loader::load("setup/llama3-8b.gguf", &model, &mut cx);
//     let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);
//     let old_logits = logits;
//     let old_cache_src = cache_src.clone();
//     let old_cache_dest = cache_dest.clone();
//     let old_input = input;
//     cx.compile(
//         (
//             GenericCompiler::default(),
//             (
//                 luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
//                 luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
//                 luminal_metal::BufferCompilers::default(),
//             ),
//         ),
//         (
//             &mut input,
//             &mut logits,
//             &mut cache_src,
//             &mut cache_dest,
//             &mut model_weights,
//         ),
//     );
//     let cache_src = downstream(&cache_src, &cx);

//     input.set_dyn(
//         input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
//         input_ids.len(),
//     );
//     cx.execute();
//     delete_inputs(&cache_src, &mut cx);
//     delete_inputs(downstream(model_weights, &cx), &mut cx);

//     let mut output_ids = vec![argmax(&logits.data())];
//     logits.drop();

//     // Decode token
//     print!("{}", cli_args.prompt.white().bold());
//     let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
//     print!("{initial}",);
//     io::stdout().flush().unwrap();

//     // Swap caches
//     transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

//     // Decode loop
//     let mut prev_output_len = initial.len();
//     for _ in 0..cli_args.gen_tokens {
//         input.set_dyn(vec![*output_ids.last().unwrap() as f32], 1);
//         cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
//         cx.execute();

//         // Sample tokens
//         let output_id = argmax(&logits.data());
//         logits.drop();
//         output_ids.push(output_id);

//         // Get the current decoded output
//         let current_output = tokenizer.decode(&output_ids, false).unwrap();

//         // Print the new substring added to the decoded output
//         print!("{}", current_output[prev_output_len..].bright_green());
//         io::stdout().flush().unwrap();

//         // Update the previous output
//         prev_output_len = current_output.len();

//         // Swap caches
//         transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
//     }

//     // let mut outputs = vec![old_to_new_mapping[&old_logits.id]];
//     // for (k, v) in &old_cache_dest {
//     //     outputs.push(old_to_new_mapping[&k.id]);
//     //     outputs.push(old_to_new_mapping[&v.id]);
//     // }
//     // println!("codegen");
//     // let kernels = codegen(
//     //     new_graph,
//     //     outputs,
//     //     luminal_2::GPUArch::Metal(HashMap::default()),
//     //     0,
//     //     &cx.dyn_map,
//     // )
//     // .unwrap();
//     // let mut inps = vec![(
//     //     old_to_new_mapping[&input.id],
//     //     Box::new(move || input_data) as Box<dyn FnOnce() -> Vec<f32>>,
//     // )];
//     // for (d, (k, v)) in cache_src_data.into_iter().zip(&old_cache_src) {
//     //     inps.push((old_to_new_mapping[&k.id], Box::new(|| d.0)));
//     //     inps.push((old_to_new_mapping[&v.id], Box::new(|| d.1)));
//     // }
//     // for (node, val) in q8_load_new("setup/llama3-8b.gguf", &model) {
//     //     inps.push((old_to_new_mapping[&node], val));
//     // }
//     // for (label, val) in accs {
//     //     match val {
//     //         InitData::Expr(e) => {
//     //             let val = e.exec(&cx.dyn_map).unwrap();
//     //             inps.push((label, Box::new(move || vec![val as f32])));
//     //         }
//     //         InitData::Data(d) => inps.push((label, Box::new(move || d))),
//     //     }
//     // }

//     // // let (buf_sizes, buf_map) = produce_buffer_map(&kernels);
//     // // println!("bufs: {:?}", buf_sizes);
//     // let (mut outputs, runtime) = run_graph(inps, &kernels, &cx.dyn_map);
//     // let logits = logits.data();
//     // println!("Old Logits: {:?}", &logits[..10]);
//     // println!("New Logits: {:?}", &outputs[0][..10]);
//     // for (i, (a, b)) in logits.iter().zip(outputs.remove(0)).enumerate() {
//     //     if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
//     //         panic!("Index {i} : {a} != {b}");
//     //     }
//     // }
//     // let mut kv_dest = cache_dest
//     //     .iter()
//     //     .rev()
//     //     .flat_map(|n| [n.0.data(), n.1.data()])
//     //     .collect_vec();
//     // let k = kv_dest.remove(0);
//     // println!("Old K: {:?}", &k[..10]);
//     // for i in 0..outputs.len() {
//     //     println!("New K: {:?}", &outputs[i][..10]);
//     // }
//     // for (i, (a, b)) in k.into_iter().zip(outputs.remove(0)).enumerate() {
//     //     if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
//     //         panic!("Index {i} : {a} != {b}");
//     //     }
//     // }
//     // let mut output_ids = vec![argmax(&logits)];
//     // // Decode token
//     // print!("{}", cli_args.prompt.white().bold());
//     // let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
//     // print!("{initial}",);
//     // io::stdout().flush().unwrap();
//     // // let v = cache_dest[0].1.data();
//     // // println!("Old V: {:?}", &v[..10]);
//     // // println!("New V: {:?}", &outputs[0][..10]);
//     // // for (i, (a, b)) in v.into_iter().zip(outputs.remove(0)).enumerate() {
//     // //     if (a - b).abs() > 1e-5 || a.is_nan() || b.is_nan() {
//     // //         panic!("Index {i} : {a} != {b}");
//     // //     }
//     // // }
//     // println!("SUCCESS!");

//     // logits.drop();
//     // transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
//     // println!("\t\t - {}ms", now.elapsed().as_millis());

//     // // Now that weights are loaded, delete the loading nodes so they don't run again
//     // delete_inputs(&cache_src, &mut cx);
//     // delete_inputs(downstream(model_weights, &cx), &mut cx);

//     // // Run prompt processing pass
//     // input.set_dyn(
//     //     input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
//     //     (1, input_ids.len()),
//     // );
//     // cx.set_dyn_dim('t', input_ids.len());
//     // print!("Processing Prompt");
//     // io::stdout().flush().unwrap();
//     // let now = Instant::now();
//     // cx.execute();
//     // let elapsed_ms = now.elapsed().as_millis();
//     // println!(
//     //     "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
//     //     1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
//     //     input_ids.len()
//     // );

//     // logits.drop();

//     // // Swap caches
//     // transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

//     // // Decode loop
//     // let start_decode = std::time::Instant::now();
//     // let mut prev_output_len = initial.len();
//     // for _ in 0..cli_args.gen_tokens {
//     //     input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
//     //     cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
//     //     cx.execute();

//     //     // Sample tokens
//     //     let output_id = argmax(&logits.data());
//     //     logits.drop();
//     //     output_ids.push(output_id);

//     //     // Get the current decoded output
//     //     let current_output = tokenizer.decode(&output_ids, false).unwrap();

//     //     // Print the new substring added to the decoded output
//     //     print!("{}", current_output[prev_output_len..].bright_green());
//     //     io::stdout().flush().unwrap();

//     //     // Update the previous output
//     //     prev_output_len = current_output.len();

//     //     // Swap caches
//     //     transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
//     // }

//     // println!();
//     // let avg_token_time =
//     //     start_decode.elapsed().as_micros() as f32 / (output_ids.len() - 1) as f32 / 1000.0;
//     // println!(
//     //     "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
//     //     avg_token_time,
//     //     1000.0 / avg_token_time
//     // );
// }

use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    time::Instant,
};

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

use crate::{
    loader::{load, load_new},
    model::KVCache,
};
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
    let initial_len = input_ids.len();

    print!("Defining graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();

    // Set up 1.0 graph
    let mut cx = Graph::new();
    let mut input = cx.named_tensor("Input", (1, 's'));
    let mut cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();
    let model = model::Llama::new(&mut cx);
    let (logits, mut cache_dest) = model.forward((input, &cache_src));
    let mut logits = logits
        .slice((.., Expression::from('s') - 1..))
        .contiguous()
        .retrieve();
    cx.set_dyn_dim('s', input_ids.len());
    cx.set_dyn_dim('p', 0);

    // Convert to 2.0 graph
    let (new_graph, old_to_new_mapping, accs) = translate_graph(&cx);
    // luminal_2::utils::display_graph(
    //     &new_graph,
    //     &[(
    //         old_to_new_mapping[&cache_dest[0].0.id],
    //         "yellow".to_string(),
    //     )],
    // );
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
    println!("OUTPUTS {}", outputs.len());

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
