mod config;
mod loader;
mod model;

use itertools::Itertools;
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
    let (out, caches) = model.forward(inp);
    out.mark();
    for (k, v) in &caches {
        k.mark_no_delete();
        v.mark_no_delete();
    }

    println!("Loading...");
    loader::DfdxDeferredLoader::new("../../Desktop/llama-dfdx-main/llama-7b-hf").load(&model, &mut cx);

    println!("Inferencing...");
    // First pass
    inp.set_dyn(input.clone(), vec![1, input.len()]);
    cx.execute();
    let out = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
    let (out_ind, _) = out[(input.len() - 1) * 32_000..].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    input.push(out_ind);
    cx.reset();

    // Build KV cache forward graph
    let (out, cache): (_, Vec<KVCache<_, usize, {config::HEADS}, {config::HEAD_DIM}>>) = model.forward_kv((inp, caches));
    let cache_ids = cache.iter().flat_map(|c| [c.0.id, c.1.id].into_iter()).collect_vec();
    cx.prune(vec![out.id], cache_ids);

    loop {
        inp.set_dyn(input.clone(), vec![1, input.len()]);
        let now = std::time::Instant::now();
        cx.execute();
        println!("Forward Pass Took {:2}s", now.elapsed().as_secs_f32());
        
        let out = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
        cx.reset();
        // Sample tokens
        let (out_ind, _) = out[(input.len() - 1) * 32_000..].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        input.push(out_ind);
        println!("{}", tokenizer.decode(input.iter().map(|i| *i as u32).collect(), false).unwrap());
    }
}
