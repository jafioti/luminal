#![allow(unused)] // WIP
use luminal::prelude::*;
use luminal_metal::MetalCompiler;

mod audio;
mod model;

fn main() {
    let mut cx = Graph::new();
    let encoder = model::AudioEncoder::initialize(&mut cx);
    let mut input = cx.tensor::<(Const<1>, Dyn<'s'>, Const<{ model::N_MEL_BINS }>)>();
    let mut encoded = encoder.forward(input);

    cx.compile(
        (<(GenericCompiler, MetalCompiler<f32>)>::default()),
        (&mut encoded, &mut input),
    );
}
