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
}
