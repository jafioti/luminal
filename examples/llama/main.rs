mod config;
mod loader;
mod model;

use luminal::prelude::*;
use model::LlamaForCausalLM;

use crate::model::KVCache;

#[rustfmt::skip]
fn main() {
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained("oobabooga/llama-tokenizer", None).unwrap();

    let mut input: Vec<usize> = tokenizer.encode("The young boy ran over to the", false).unwrap().get_ids().iter().map(|i| *i as usize).collect();

    println!("Creating Graph...");
    let mut cx = Graph::new();
    let model: LlamaForCausalLM<
        { config::VOCAB },
        { config::HEADS },
        { config::HIDDEN },
        { config::INTERMEDIATE },
        { config::HEAD_DIM },
        { config::HEAD_DIM_OVER_2 },
        { config::LAYERS },
    > = InitModule::initialize(&mut cx);
    let inp = cx.new_tensor::<(usize, usize)>("Input");
    let (out, cache_src) = model.forward(inp);
    out.mark();
    for (k, v) in &cache_src {
        k.mark_no_delete();
        v.mark_no_delete();
    }

    println!("Loading...");
    loader::DfdxDeferredLoader::new("../../Desktop/llama-dfdx-main/llama-7b-hf").load(&model, &mut cx);

    println!("Inferencing...");
    // First pass
    inp.set_dyn(input.clone(), vec![1, input.len()]);
    let now = std::time::Instant::now();
    cx.execute();
    println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());

    let out = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
    input.push(sample_index(&out[(input.len() - 1) * 32_000..]));
    println!("{}", tokenizer.decode(input.iter().map(|i| *i as u32).collect(), false).unwrap());


    // Build KV cache forward graph
    let (out, cache_dest): (_, Vec<KVCache<_, usize, {config::HEADS}, {config::HEAD_DIM}>>) = model.forward_kv((inp, cache_src.clone()));
    out.mark();
    for (k, v) in &cache_dest {
        k.mark_no_delete();
        v.mark_no_delete();
    }
    cx.prune([out.id], cache_src.iter().flat_map(|(k, v)| [k.id, v.id]));

    loop {
        inp.set_dyn(vec![*input.last().unwrap()], vec![1, 1]);

        let now = std::time::Instant::now();
        cx.execute();
        println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());
        
        let o = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
        // Sample tokens
        input.push(sample_index(&o));
        println!("{}", tokenizer.decode(input.iter().map(|i| *i as u32).collect(), false).unwrap());

        // Swap caches
        for ((src_k, src_v), (dest_k, dest_v)) in cache_src.iter().copied().zip(cache_dest.iter().copied()) {
            // Move dest caches to src
            cx.swap_tensors(src_k, dest_k);
            cx.swap_tensors(src_v, dest_v);
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
