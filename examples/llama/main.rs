mod config;
mod loader;
mod model;

use luminal::prelude::*;
use model::LlamaForCausalLM;

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
    let (out, _caches) = model.forward((inp, 0));
    inp.set_dyn(vec![1, 2, 3], vec![1, 3]);
    out.mark();

    println!("Loading...");
    loader::DfdxDeferredLoader::new("../../Desktop/llama-dfdx-main/llama-7b-hf").load(&model, &mut cx);

    println!("Inferencing...");
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
