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
    let inp = cx.new_tensor::<(usize, Const<64>)>();
    let out = model.forward((inp, 0));
    inp.set_dyn(vec![0; 2 * 64], vec![2, 64]);
    out.mark();
    cx.execute();
    model.save_to_file(&cx, "model.st");

    println!(
        "Out: {:?}",
        out.retrieve()
            .unwrap()
            .real_data(out.view().unwrap())
            .unwrap()
    );
}
