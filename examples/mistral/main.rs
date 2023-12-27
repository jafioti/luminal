use crate::model::Mistral;
use itertools::Itertools;
mod model;

fn main() -> Result<(), String> {
    println!("Defining the model and loading the tokenizer");
    let mut mistral = Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model")
        .map_err(|e| e.to_string())?;

    static FILE_PATHS: [&'static str; 3] = [
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
    ];

    println!("Loading the model weights from safetensors");
    unsafe {
        mistral.load_safe_tensors_from_files(&FILE_PATHS)?;
    }

    // Test inference
    let prompt = "Santa says: Merry";

    mistral.debug_run(prompt);

    // // Build the forward graph
    // println!("Building the forward graph");
    // let output_probabilities = mistral.build_forward_graph(prompt);

    // Compile the graph
    // println!("Compiling the forward graph");
    // mistral.compile_forward_graph();

    // println!("Infering the next token");
    // let _ = mistral.infer_next_token(output_probabilities, prompt);

    // println!("Inference: {output_text}");

    Ok(())
}
