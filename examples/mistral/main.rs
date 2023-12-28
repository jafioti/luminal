use colored::Colorize;
use itertools::Itertools;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};
mod model;

use luminal::{prelude::*, shape::symbolic::Expression};

fn main() -> Result<(), String> {
    println!("Constructing graph...");
    let tokenizer = SentencePieceBpeTokenizer::from_file(
        "./examples/mistral/setup/mistral-7b-hf/tokenizer.model",
        false,
    )
    .unwrap();

    let file_paths = vec![
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors".to_string(),
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors".to_string(),
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors".to_string(),
    ];

    let mut cx = Graph::new();
    let input = cx.named_tensor::<(Const<1>, Dyn<'s'>)>("Input");
    let model = model::MistralLM::initialize(&mut cx);
    let logits = model
        .forward(input)
        .slice((.., (Expression::from('s') - 1).., ..))
        .retrieve();
    SafeTensorLoader::new(file_paths).load(&model, &mut cx);

    println!("Compiling graph...");
    cx.compile(GenericCompiler::<MetalFp16Compiler>::default());
    // Cache model weights
    cx.compile(RemapDownstream(
        state_dict(&model).values().copied().collect(),
    ));
    keep_weights(&model, &mut cx);

    // Initial forward pass to load weights
    println!("Loading model...");
    input.set_dyn(vec![1.], vec![1, 1]);
    cx.execute();
    logits.drop();
    // Now that weights are loaded, delete the loading nodes so they don't run again
    delete_inputs(
        &state_dict(&model).values().copied().collect::<Vec<_>>(),
        &mut cx,
    );

    // Run inference
    let prompt = "Santa says: Merry";
    let input_data = encode(&tokenizer, prompt);

    let mut context_vector = input_data;
    let mut completion = String::new();
    let max_new_tokens = 40;
    for i in 0..max_new_tokens {
        println!("########################### Iteration {i} ###########################");
        input.set_dyn(context_vector.clone(), vec![1, context_vector.len()]);
        cx.execute();

        let output_id = sample_index(&logits.data());
        context_vector.push(output_id as f32);

        // Decode token
        let output_token = decode(&tokenizer, vec![output_id as f32]);
        completion.push_str(&output_token);

        println!("{}{}", prompt.on_black().white().bold(), completion.green());

        logits.drop();
    }

    Ok(())
}

// Method to encode text as vector
pub fn encode(tokenizer: &SentencePieceBpeTokenizer, text: &str) -> Vec<f32> {
    let mut vector = tokenizer
        .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
        .token_ids
        .iter()
        .map(|&x| x as f32)
        .collect_vec();

    vector.insert(0, 1.0); // Start token

    vector
}

pub fn decode(tokenizer: &SentencePieceBpeTokenizer, token_ids: Vec<f32>) -> String {
    let binding = token_ids.iter().map(|i| *i as i64).collect_vec();
    let token_ids = binding.as_slice();

    tokenizer
        .decode(token_ids, true, false)
        .replace("<0x0A>", "\n")
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> usize {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}
