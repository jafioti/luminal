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
    unsafe {
        mistral
            .load_safe_tensors_from_files(file_paths.iter().map(|s| s.to_string()).collect_vec())?;
    }

    // Build the forward graph
    println!("Building the forward graph");
    let output_token_ids = mistral.build_forward_graph();

    // Compile the graph
    println!("Compiling the forward graph");
    mistral.compile_forward_graph();

    // Test inference
    let input_text = "Hello, how are";
    println!("Infering the next token");
    let output_text = mistral.infer_next_token(output_token_ids, input_text);

    println!("Inference: {output_text}");

    Ok(())
}
