mod config;
mod model;

use luminal::prelude::*;
use model::Llama;

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
    let out = model.forward((inp, 0));
    inp.set_dyn(vec![0; 5], vec![1, 5]);
    out.mark();

    cx.optimize(GenericOptimizer::default());
    cx.execute();
}
