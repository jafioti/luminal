use std::{marker::PhantomData, time::Instant};

use colored::Colorize;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};
mod model;

use luminal::{prelude::*, shape::symbolic::Expression};

use crate::model::KVCache;

#[cfg(feature = "metal")]
type DeviceCompiler = MetalFp16Compiler;
#[cfg(feature = "cuda")]
type DeviceCompiler = CudaFp16Compiler;
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
type DeviceCompiler = CPUCompiler;

fn main() -> Result<(), String> {
    println!("Constructing graph...");
    let tokenizer = SentencePieceBpeTokenizer::from_file(
        "./examples/mistral/setup/mistral-7b-hf/tokenizer.model",
        false,
    )
    .unwrap();

    let mut cx1 = Graph::new();
    let input = cx1.named_tensor::<(Const<1>, Dyn<'s'>)>("Input");
    let model = model::MistralLM::initialize(&mut cx1);
    let (logits, kv_cache) = model.forward((
        input,
        Option::<Vec<KVCache<Const<1>, Const<0>>>>::None,
        PhantomData::<Dyn<'s'>>,
    ));
    let logits = logits
        .slice((.., (Expression::from('s') - 1).., ..))
        .retrieve();
    kv_cache.keep();
    SafeTensorLoader::new(vec![
        "./examples/mistral/setup/mistral-7b-hf/model-00001-of-00003.safetensors".to_string(),
        "./examples/mistral/setup/mistral-7b-hf/model-00002-of-00003.safetensors".to_string(),
        "./examples/mistral/setup/mistral-7b-hf/model-00003-of-00003.safetensors".to_string(),
    ])
    .load(&model, &mut cx1);
    let mut cx2 = Graph::new();
    let single_input = cx2.named_tensor::<R2<1, 1>>("Input");
    let kv_model = model::MistralLM::initialize(&mut cx2);
    let cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx2.named_tensor("Key Cache"),
                cx2.named_tensor("Value Cache"),
            )
        })
        .collect();
    let (decode_logits, cache_dest) = kv_model.forward((
        single_input,
        Some(cache_src.clone()),
        PhantomData::<Dyn<'t'>>,
    ));
    decode_logits.retrieve();
    cache_dest.keep();

    println!("Compiling graph...");
    cx1.compile(GenericCompiler::<DeviceCompiler>::default());
    // Cache model weights
    cx1.compile(RemapDownstream(
        state_dict(&model).values().copied().collect(),
    ));
    keep_weights(&model, &mut cx1);

    // Compile second graph
    cx2.compile(GenericCompiler::<DeviceCompiler>::default());
    // Cache model weights
    cx2.compile(RemapDownstream(
        state_dict(&kv_model).values().copied().collect(),
    ));
    keep_weights(&kv_model, &mut cx2);
    delete_inputs(
        &state_dict(&kv_model).values().copied().collect::<Vec<_>>(),
        &mut cx2,
    );
    delete_inputs(
        &cache_src
            .iter()
            .flat_map(|(k, v)| [k.id(), v.id()])
            .collect::<Vec<_>>(),
        &mut cx2,
    );

    // Initial forward pass to load weights
    println!("Loading model...");
    input.set_dyn(vec![1.], vec![1, 1]);
    cx1.execute();
    logits.drop();
    kv_cache.drop();

    // Now that weights are loaded, delete the loading nodes so they don't run again
    delete_inputs(
        &state_dict(&model).values().copied().collect::<Vec<_>>(),
        &mut cx1,
    );

    // Run inference first pass
    let prompt = "Santa says: Merry";
    let mut input_ids = encode(&tokenizer, prompt);

    let mut completion = String::new();
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        vec![1, input_ids.len()],
    );
    cx1.execute();

    let output_id = sample_index(&logits.data());
    input_ids.push(output_id);

    // Decode token
    completion.push_str(&decode(&tokenizer, &[output_id]));
    println!("{}{}", prompt.on_black().white().bold(), completion.green());

    // Transfer weights and kv cache
    transfer_weights(&model, &mut cx1, &kv_model, &mut cx2);
    for ((key_src, val_src), (key_dest, val_dest)) in kv_cache.into_iter().zip(cache_src.iter()) {
        cx2.set_tensor(key_dest.id(), 0, cx1.get_tensor(key_src.id(), 0).unwrap());
        cx2.set_tensor(val_dest.id(), 0, cx1.get_tensor(val_src.id(), 0).unwrap());
    }

    // Decode loop
    for _ in 0..100 {
        single_input.set(vec![*input_ids.last().unwrap() as f32]);
        cx2.set_dyn_dim('p', input_ids.len() - 1);
        cx2.set_dyn_dim('t', input_ids.len());

        let now = Instant::now();
        cx2.execute();
        println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());

        // Sample tokens
        let output_id = sample_index(&decode_logits.data());
        decode_logits.drop();
        completion.push_str(&decode(&tokenizer, &[output_id]));
        input_ids.push(output_id);
        println!("{}{}", prompt.on_black().white().bold(), completion.green());

        // Swap caches
        for ((src_k, src_v), (dest_k, dest_v)) in cache_src.iter().zip(cache_dest.iter()) {
            // Move dest caches to src
            cx2.swap_tensors(*src_k, *dest_k);
            cx2.swap_tensors(*src_v, *dest_v);
            // // Drop dest caches
            // dest_k.drop();
            // dest_v.drop();
        }
    }

    Ok(())
}

// Method to encode text as vector
pub fn encode(tokenizer: &SentencePieceBpeTokenizer, text: &str) -> Vec<i64> {
    let mut vector = tokenizer
        .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
        .token_ids;
    vector.insert(0, 1); // Start token
    vector
}

pub fn decode(tokenizer: &SentencePieceBpeTokenizer, token_ids: &[i64]) -> String {
    tokenizer
        .decode(token_ids, true, false)
        .replace("<0x0A>", "\n")
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> i64 {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0 as i64
}
