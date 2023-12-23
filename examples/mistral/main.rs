use std::{borrow::Cow, collections::HashMap, fs::File};

use crate::model::{compute_rotary_embedding_frequencies, Mistral};
use itertools::Itertools;
use luminal::prelude::*;
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use yoke::{Yoke, Yokeable};

mod model;

#[derive(Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

fn main() {
    let mut graph = Graph::new();

    // Let's figure out argmax
    let x1 = graph.tensor::<R2<2, 5>>();
    x1.set(vec![
        0.0, 99.0, 100.0, 101.0, -5.0, 0.0, 0.0, 6.0, 2.0, -5.0,
    ]);
    x1.retrieve();
    // let x1_max = x1.max_reduce::<R1<1>, Axis<1>>().expand::<R2<1, 10>, _>();
    // // x1_max.retrieve();

    // let x1_equal = x1.equals(x1_max);
    // // x1_equal.retrieve();

    // // Goal is to find the arg max
    // let r = graph.arange::<Const<10>>();

    // let y = x1_equal * r.expand();
    // let y = y.max_reduce::<R1<1>, Axis<1>>();

    // let y = graph.arange_::<R3<2, 3, 5>>();

    let y = x1.argmax();
    y.retrieve();

    graph.execute();

    println!("{:?}", x1);
    println!("Argmax: {:?}", y);
}

// fn main() -> Result<(), String> {
//     // // A range with a step of 2
//     // let mut graph = Graph::new();

//     // // let r = graph.arange::<Const<10>>();
//     // // let r = r.pow2(2.0);

//     // let (r, i) = compute_rotary_embedding_frequencies::<Const<6>, 8>(&mut graph);
//     // r.retrieve();
//     // i.retrieve();

//     // graph.compile(<(PreGenericCompiler, MetalFp32Compiler, PostGenericCompiler)>::default());

//     // graph.execute();

//     // println!("{:?}", r);

//     let mut mistral =
//         Mistral::new("./examples/mistral/setup/mistral-7b-hf/tokenizer.model").unwrap();
//     // let filename = "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors";

//     let file_paths = [
//         "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors",
//         "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors",
//         "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors",
//     ];

//     unsafe {
//         mistral
//             .load_safe_tensors_from_files(file_paths.iter().map(|s| s.to_string()).collect_vec())?;
//     }

//     mistral
//         .graph
//         .compile(<(PreGenericCompiler, MetalFp16Compiler, PostGenericCompiler)>::default());

//     // mistral.graph.execute_debug();

//     let input_text = "Hello, how are";
//     let output_text = mistral.infer_next_token(input_text);

//     println!("Inference: {output_text}");

//     Ok(())
// }
