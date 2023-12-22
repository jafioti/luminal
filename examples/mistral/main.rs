use std::{borrow::Cow, collections::HashMap, fs::File};

use crate::model::{pre_compute_rotary_embedding_frequencies, Mistral};
use itertools::Itertools;
use luminal::prelude::*;
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use yoke::{Yoke, Yokeable};

mod model;

#[derive(Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

fn main() -> Result<(), String> {
    // A range with a step of 2
    let mut graph = Graph::new();

    // let r = graph.arange::<Const<10>>();
    // let r = r.pow2(2.0);

    let (r, i) = pre_compute_rotary_embedding_frequencies::<8, Const<6>>(&mut graph);
    r.retrieve();
    i.retrieve();

    graph.compile(<(PreGenericCompiler, MetalFp32Compiler, PostGenericCompiler)>::default());

    graph.execute();

    println!("{:?}", r);

    // let mut mistral =
    //     Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model").unwrap();
    // // let filename = "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors";

    // let file_paths = [
    //     "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
    //     "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
    //     "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
    // ];

    // unsafe {
    //     mistral
    //         .load_safe_tensors_from_files(file_paths.iter().map(|s| s.to_string()).collect_vec())?;
    // }
    // mistral.embedding.retrieve();
    // mistral
    //     .graph
    //     .compile(<(PreGenericCompiler, MetalFp16Compiler, PostGenericCompiler)>::default());

    // mistral.graph.execute_debug();

    // println!(
    //     "{:?}",
    //     mistral
    //         .embedding
    //         .data()
    //         .iter()
    //         .skip(1000000)
    //         .take(30)
    //         .collect_vec()
    // );

    Ok(())
}
