use crate::model::Mistral;
use itertools::Itertools;
use luminal::prelude::*;

mod model;

fn main() -> Result<(), String> {
    let mut mistral =
        Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model").unwrap();
    let filename = "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors";
    mistral.load_safe_tensors_from_file(&filename)?;
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
