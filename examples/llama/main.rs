mod config;
mod loader;
mod model;

use cudarc::driver::CudaSlice;
use half::f16;
use luminal::prelude::*;
use model::LlamaForCausalLM;
use petgraph::stable_graph::NodeIndex;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

type Model = LlamaForCausalLM<
    { config::VOCAB },
    { config::HEADS },
    { config::HIDDEN },
    { config::INTERMEDIATE },
    { config::HEAD_DIM },
    { config::HEAD_DIM_OVER_2 },
    { config::LAYERS },
>;

#[rustfmt::skip]
fn main() {
    let prompt = "Here is a python implementation of merge sort:";
    let tokenizer =
            SentencePieceBpeTokenizer::from_file("./examples/llama/setup/llama-7b-hf/tokenizer.model", false).unwrap();
    let mut input = tokenizer.encode(
        prompt,
        None,
        prompt.len(),
        &TruncationStrategy::LongestFirst,
        0
    ).token_ids.iter().map(|&x| x as usize).collect::<Vec<_>>();
    input.insert(0, 1); // Start token

    println!("Creating Graphs...");
    let mut cx1 = Graph::new();
    let mut cx2 = Graph::new();
    let model = Model::initialize(&mut cx1);
    // mark_weights(&model, &mut cx1);
    let inp = cx1.new_tensor::<(Dyn<'b'>, Dyn<'s'>)>("Input");
    let (out1, cache1) = model.forward(inp);
    out1.mark();
    loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx1);
    for (k, v) in &cache1 {
        k.mark_no_delete();
        v.mark_no_delete();
    }
    cx1.optimize(<(MetalFp16Optimizer, GenericOptimizer)>::default());
    // cx1.optimize(<(MetalFp32Optimizer, GenericOptimizer)>::default());
    // cx1.optimize(<(CPUOptimizer, GenericOptimizer)>::default());

    // Build KV cache forward graph
    let kv_model = Model::initialize(&mut cx2);
    // mark_weights(&kv_model, &mut cx2);
    let single_inp = cx2.new_tensor::<(Dyn<'b'>, Const<1>)>("Input");
    let cache_src = (0..config::LAYERS).map(|_| (cx2.new_tensor("Key Cache"), cx2.new_tensor("Value Cache"))).collect::<Vec<_>>();
    let (out, cache_dest)= kv_model.forward_kv::<_, _, Dyn<'p'>, Dyn<'t'>>((single_inp, cache_src.clone()));
    out.mark();
    loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&kv_model, &mut cx2);
    for (k, v) in &cache_dest {
        k.mark_no_delete();
        v.mark_no_delete();
    }
    for (k, v) in &cache_src {
        k.set_type(std::any::TypeId::of::<CudaSlice<f16>>());
        v.set_type(std::any::TypeId::of::<CudaSlice<f16>>());
    }
    cx2.optimize(<(MetalFp16Optimizer, GenericOptimizer)>::default());
    // cx2.optimize(<(CPUOptimizer, GenericOptimizer)>::default());

    println!("Inferencing...");
    // First pass
    inp.set_dyn(input.iter().map(|i| *i as f32).collect::<Vec<_>>(), vec![1, input.len()]);
    cx1.execute_debug();

    let out1 = out1.dyn_data(&cx1.dyn_map);
    input.push(sample_index(&out1[out1.len() - 32_000..]));
    println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false).replace("<0x0A>", "\n"));

    // Move cache over to second graph
    for ((key_src, val_src), (key_dest, val_dest)) in cache1.into_iter().zip(cache_src.iter()) {
        cx2.set_tensor(key_dest.id(), 0, cx1.get_tensor(key_src.id(), 0).unwrap());
        cx2.set_tensor(val_dest.id(), 0, cx1.get_tensor(val_src.id(), 0).unwrap());
        key_dest.mark_no_delete();
        val_dest.mark_no_delete();
    }

    // Move weights over
    transfer_weights(&model, &mut cx1, &kv_model, &mut cx2);
    drop(cx1);

    loop {
        single_inp.set_dyn(vec![*input.last().unwrap() as f32], vec![1, 1]);
        cx2.set_dyn_dim('p', input.len() - 1);
        cx2.set_dyn_dim('t', input.len());

        let now = std::time::Instant::now();
        cx2.execute();
        println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());
        
        let o = out.dyn_data(&cx2.dyn_map);
        out.drop();
        // Sample tokens
        input.push(sample_index(&o));
        println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false).replace("<0x0A>", "\n"));

        // Swap caches
        for ((src_k, src_v), (dest_k, dest_v)) in cache_src.iter().copied().zip(cache_dest.iter().copied()) {
            // Move dest caches to src
            cx2.swap_tensors(src_k, dest_k);
            cx2.swap_tensors(src_v, dest_v);
            // Drop dest caches
            dest_k.drop();
            dest_v.drop();
        }
    }
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> usize {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}
