mod loader;
mod model;

use std::{io::Write, marker::PhantomData, time::Instant};

use colored::Colorize;
use luminal::{prelude::*, shape::symbolic::Expression};
use model::LlamaForCausalLM;
use rust_tokenizers::tokenizer::{
    SentencePieceBpeTokenizer, Tokenizer,
    TruncationStrategy::{self},
};

use crate::model::KVCache;

type Model = LlamaForCausalLM<
    { model::VOCAB },
    { model::HEADS },
    { model::HIDDEN },
    { model::INTERMEDIATE },
    { model::HEAD_DIM },
    { model::HEAD_DIM_OVER_2 },
    { model::LAYERS },
>;

#[cfg(feature = "metal")]
type DeviceCompiler = MetalFp16Compiler;
#[cfg(feature = "cuda")]
type DeviceCompiler = CudaFp16Compiler;
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
type DeviceCompiler = CPUCompiler;

fn main() {
    let prompt = "Here is a python implementation of merge sort:";
    let tokenizer = SentencePieceBpeTokenizer::from_file(
        "./examples/llama/setup/llama-7b-hf/tokenizer.model",
        false,
    )
    .unwrap();
    let mut input = tokenizer
        .encode(
            prompt,
            None,
            prompt.len(),
            &TruncationStrategy::LongestFirst,
            0,
        )
        .token_ids
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();
    input.insert(0, 1); // Start token

    println!("Creating Graphs...");
    let mut cx1 = Graph::new(); // Prompt processing graph
    let mut cx2 = Graph::new(); // Token generation graph
    let model = Model::initialize(&mut cx1);
    let inp = cx1.named_tensor::<(Const<1>, Dyn<'s'>)>("Input").set_dyn(
        input.iter().map(|i| *i as f32).collect::<Vec<f32>>(),
        vec![1, input.len()],
    );
    let (out1, cache1) = model.forward((
        inp,
        Option::<Vec<KVCache<Const<1>, Const<0>, { model::HEADS }, { model::HEAD_DIM }>>>::None,
        PhantomData::<Dyn<'s'>>,
    ));
    out1.retrieve();
    cache1.keep();
    loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx1);

    cx1.compile(GenericCompiler::<DeviceCompiler>::default());

    // Cache model weights
    cx1.compile(RemapDownstream(
        state_dict(&model).values().copied().collect(),
    ));
    keep_weights(&model, &mut cx1);

    // Build KV cache forward graph
    let kv_model = Model::initialize(&mut cx2);
    let single_inp = cx2.named_tensor::<R2<1, 1>>("Input");
    let cache_src: Vec<KVCache<Const<1>, Dyn<'p'>, { model::HEADS }, { model::HEAD_DIM }>> = (0
        ..model::LAYERS)
        .map(|_| {
            (
                cx2.named_tensor("Key Cache"),
                cx2.named_tensor("Value Cache"),
            )
        })
        .collect();
    let (out, cache_dest) =
        kv_model.forward((single_inp, Some(cache_src.clone()), PhantomData::<Dyn<'t'>>));
    out.retrieve();
    cache_dest.keep();
    cx2.compile(GenericCompiler::<DeviceCompiler>::default());

    // Cache model weights
    cx2.compile(RemapDownstream(
        state_dict(&kv_model).values().copied().collect(),
    ));
    keep_weights(&kv_model, &mut cx2);
    // Delete weight loading nodes
    delete_inputs(
        &state_dict(&kv_model).values().copied().collect::<Vec<_>>(),
        &mut cx2,
    );
    // Delete cache loading nodes
    delete_inputs(
        &cache_src
            .iter()
            .flat_map(|(k, v)| [k.id(), v.id()])
            .collect::<Vec<_>>(),
        &mut cx2,
    );

    println!("Inferencing...");
    // First pass
    cx1.execute_debug();

    let out1 = out1.data();
    input.push(sample_index(&out1[out1.len() - 32_000..]) as usize);
    println!(
        "{}",
        tokenizer
            .decode(
                &input.iter().map(|i| *i as i64).collect::<Vec<_>>(),
                true,
                false
            )
            .replace("<0x0A>", "\n")
    );

    // Move cache over to second graph
    for ((key_src, val_src), (key_dest, val_dest)) in cache1.into_iter().zip(cache_src.iter()) {
        cx2.set_tensor(key_dest.id(), 0, cx1.get_tensor(key_src.id(), 0).unwrap());
        cx2.set_tensor(val_dest.id(), 0, cx1.get_tensor(val_src.id(), 0).unwrap());
    }

    // Move weights over
    transfer_weights(&model, &mut cx1, &kv_model, &mut cx2);

    loop {
        single_inp.set(vec![*input.last().unwrap() as f32]);
        cx2.set_dyn_dim('p', input.len() - 1);
        cx2.set_dyn_dim('t', input.len());

        let now = Instant::now();
        cx2.execute();
        println!("Forward Pass Took {}ms", now.elapsed().as_millis());

        let o = out.data();
        out.drop();
        // Sample tokens
        input.push(sample_index(&o) as usize);
        println!(
            "{}",
            tokenizer
                .decode(
                    &input.iter().map(|i| *i as i64).collect::<Vec<_>>(),
                    true,
                    false
                )
                .replace("<0x0A>", "\n")
        );

        // Swap caches
        for ((src_k, src_v), (dest_k, dest_v)) in cache_src.iter().zip(cache_dest.iter()) {
            // Move dest caches to src
            cx2.swap_tensors(*src_k, *dest_k);
            cx2.swap_tensors(*src_v, *dest_v);
            // Drop dest caches
            dest_k.drop();
            dest_v.drop();
        }
    }
}

fn encode(tokenizer: &SentencePieceBpeTokenizer, text: &str) -> Vec<i64> {
    let mut vector = tokenizer
        .encode(text, None, text.len(), &TruncationStrategy::LongestFirst, 0)
        .token_ids;
    vector.insert(0, 1); // Start token
    vector
}

fn decode(tokenizer: &SentencePieceBpeTokenizer, token_ids: &[i64]) -> String {
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
