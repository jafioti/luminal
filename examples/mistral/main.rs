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

    let filepaths = [
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
    ];

    let mut variable_file_mapper = HashMap::new();
    let mut file_tensor_mapper = HashMap::new();

    for filepath in filepaths {
        let file = File::open(filepath).map_err(|e| e.to_string())?;
        let file = unsafe { MmapOptions::new().map(&file).map_err(|e| e.to_string())? };
        // let tensors = SafeTensors::deserialize(&buffer).map_err(|e| e.to_string())?;
        let tensors = Yoke::<SafeTensors_<'static>, Mmap>::try_attach_to_cart(file, |data| {
            let tensors = SafeTensors::deserialize(data).map_err(|e| e.to_string())?;
            Ok::<_, String>(SafeTensors_(tensors))
        })?;

        for name in tensors.get().0.names() {
            variable_file_mapper.insert(name.to_string(), filepath.to_string());
        }

        file_tensor_mapper.insert(filepath.to_string(), tensors);
    }

    println!("{:?}", variable_file_mapper);

    // mistral.load_safe_tensors_from_file(&filename)?;
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
