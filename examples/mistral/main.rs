use crate::model::Mistral;
use itertools::Itertools;
use luminal::{
    compilers::{MetalFp16Compiler, PostGenericCompiler, PreGenericCompiler},
    graph::Graph,
    graph_tensor::GraphTensor,
    shape::R2,
};
mod model;

// fn main() {
//     let mut graph = Graph::new();

//     let x = graph.arange_::<R2<4, 8>>();
//     x.retrieve();

//     // let y = x.slice((.., ..6));
//     // let y = x
//     //     .reshape::<R2<4, 6>>()
//     //     .pad::<R2<4, 8>, usize, usize>(&[(0, 0), (0, 2)]);
//     let mut y = x.contiguous();
//     y.shape.slice(&[(0.into(), 4.into()), (0.into(), 6.into())]);
//     y.retrieve();

//     graph.compile(<(
//         PreGenericCompiler,
//         MetalFp16Compiler,
//         // CPUCompiler,
//         PostGenericCompiler,
//     )>::default());

//     graph.execute();

//     println!("x: {:?}", x);
//     println!("y: {:?}", y);
// }

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

    // mistral.debug_run();

    // Test inference
    let prompt = "Merry ";

    // Build the forward graph
    println!("Building the forward graph");
    let output_token_ids = mistral.build_forward_graph(prompt);

    // Compile the graph
    println!("Compiling the forward graph");
    mistral.compile_forward_graph();

    println!("Infering the next token");
    let output_text = mistral.infer_next_token(output_token_ids, prompt);

    println!("Inference: {output_text}");

    Ok(())
}
