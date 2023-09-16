mod config;
mod loader;
mod model;

use luminal::prelude::*;
use model::LlamaForCausalLM;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

#[rustfmt::skip]
fn main() {
    let prompt = "The young boy ran over to the";
    let tokenizer =
            SentencePieceBpeTokenizer::from_file("./examples/llama/setup/llama-7b-hf/tokenizer.model", false).unwrap();
    let mut input: Vec<usize> = tokenizer.encode(
        prompt,
        None,
        prompt.len(),
        &TruncationStrategy::LongestFirst,
        0
    ).token_ids.iter() .map(|&x| x as usize).collect();
    input.insert(0, 1);

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
    let mut inp = cx.new_tensor::<(Dyn<'b'>, Dyn<'s'>)>("Input");
    let (out, cache_src) = model.forward(inp);
    let cache_src = cache_src.into_iter().map(|(a, b)| {
        (a.reshape(), b.reshape())
    }).collect::<Vec<_>>();
    out.mark();
    for (k, v) in &cache_src {
        k.mark_no_delete();
        v.mark_no_delete();
    }

    println!("Loading...");
    loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx);

    println!("Inferencing...");
    // First pass
    inp.set_dyn(input.clone(), vec![1, input.len()]);
    let now = std::time::Instant::now();

    cx.execute();
    println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());

    let out = out.dyn_data(&cx.dyn_map);
    input.push(sample_index(&out[out.len() - 32_000..]));
    println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false));


    // Build KV cache forward graph
    let mut single_inp = cx.new_tensor::<(Dyn<'b'>, Const<1>)>("Single Input");
    let (out, cache_dest)= model.forward_kv::<_, _, Dyn<'p'>, Dyn<'t'>>((single_inp, cache_src.clone()));
    out.mark();
    for (k, v) in &cache_dest {
        k.mark_no_delete();
        v.mark_no_delete();
    }
    cx.prune([out.id], cache_src.iter().flat_map(|(k, v)| [k.id, v.id]));

    loop {
        single_inp.set_dyn(vec![*input.last().unwrap()], vec![1, 1]);
        cx.set_dyn_dim('p', input.len() - 1);
        cx.set_dyn_dim('t', input.len());

        let now = std::time::Instant::now();
        cx.execute();
        println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());
        
        let o = out.dyn_data(&cx.dyn_map);
        out.drop();
        // Sample tokens
        input.push(sample_index(&o));
        println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false));

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
