use crate::model::Mistral;
use itertools::Itertools;
mod model;

fn main() -> Result<(), String> {
    println!("Defining the model and loading the tokenizer");
    let mut mistral = Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model")
        .map_err(|e| e.to_string())?;

    let file_paths = [
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
    ];

    println!("Loading the model weights from safetensors");
    mistral.load(file_paths.iter().map(|s| s.to_string()).collect_vec())?;

    // Test inference
    let prompt = "Santa says: Merry";

    mistral.debug_run(prompt);

    Ok(())
}
