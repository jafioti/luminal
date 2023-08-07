mod config;
mod loader;
mod model;

use luminal::prelude::*;
use model::Llama;

#[rustfmt::skip]
fn main() {
    let mut cx = Graph::new();
    let model: Llama<
        { config::VOCAB },
        { config::HEADS },
        { config::HIDDEN },
        { config::INTERMEDIATE },
        { config::HEAD_DIM },
        { config::HEAD_DIM_OVER_2 },
        { config::LAYERS },
    > = InitModule::initialize(&mut cx);
    let inp = cx.new_tensor::<(usize, usize)>("Input");
    let (out, caches) = model.forward((inp, 0));
    inp.set_dyn(vec![1, 2, 3], vec![1, 3]);
    out.mark();

    loader::DfdxDeferredLoader::new("../../Desktop/llama-dfdx-main/llama-7b-hf/model").load(&model, &mut cx);

    cx.execute();
    std::fs::write("out.bin", out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap().into_iter().flat_map(|i| i.to_le_bytes().into_iter()).collect::<Vec<u8>>()).unwrap();
    // println!("Out: {:?}", out.retrieve().unwrap().real_data(out.view().unwrap()).unwrap());
}
