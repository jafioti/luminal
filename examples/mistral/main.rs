use std::{borrow::Cow, collections::HashMap, fs::File};

use crate::model::Mistral;
use itertools::Itertools;
use luminal::prelude::*;
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use yoke::{Yoke, Yokeable};

mod model;

#[derive(Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

fn main() -> Result<(), String> {
    let mut mistral =
        Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model").unwrap();
    // let filename = "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors";

    let file_paths = [
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
    ];

    unsafe {
        mistral
            .load_safe_tensors_from_files(file_paths.iter().map(|s| s.to_string()).collect_vec())?;
    }
    mistral.embedding.retrieve();
    mistral
        .graph
        .compile(<(PreGenericCompiler, MetalFp16Compiler, PostGenericCompiler)>::default());

    mistral.graph.execute_debug();

    println!(
        "{:?}",
        mistral
            .embedding
            .data()
            .iter()
            .skip(1000000)
            .take(30)
            .collect_vec()
    );

    Ok(())
}
