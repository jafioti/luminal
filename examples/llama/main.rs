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
    cx.no_delete.remove(&out.id);
    let out = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
    cx.reset();
    let (out_ind, _) = out[(input.len() - 1) * 32_000..].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    input.push(out_ind);
    println!("{}", tokenizer.decode(input.iter().map(|i| *i as u32).collect(), false).unwrap());


    // Build KV cache forward graph
    let inp = cx.new_tensor::<(usize, usize)>("NewInput");
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
        cx.reset();
        
        let o = out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap();
        cx.tensors.remove(&cx.views.remove(&out.id).unwrap().tensor_id);
        // Sample tokens
        let (out_ind, _) = o.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        input.push(out_ind);
        println!("{}", tokenizer.decode(input.iter().map(|i| *i as u32).collect(), false).unwrap());

        // Swap caches
        for ((a_k, a_v), (b_k, b_v)) in cache_src.iter().copied().zip(cache_dest.iter().copied()) {
            // Remove source caches
            cx.swap_tensors(a_k, b_k);
            cx.swap_tensors(a_v, b_v);
            cx.tensors.remove(&cx.views.remove(&b_k.id).unwrap().tensor_id);
            cx.tensors.remove(&cx.views.remove(&b_v.id).unwrap().tensor_id);
        }
    }
}
