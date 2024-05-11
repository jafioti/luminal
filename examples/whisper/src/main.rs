#![allow(unused)]
use std::marker::PhantomData;

// WIP
use luminal::prelude::*;
use luminal_metal::MetalCompiler;
use model::KVCache;

mod audio;
mod gguf;
mod loader;
mod model;

fn main() {
    // Construct encoder graph
    let mut enc_cx = Graph::new();
    let encoder = model::AudioEncoder::initialize(&mut enc_cx);
    let mut audio_input = enc_cx.tensor::<(Const<1>, Dyn<'s'>, Const<{ model::N_MEL_BINS }>)>();
    let mut encoded = encoder.forward(audio_input).retrieve();
    loader::load("setup/whisper.gguf", &encoder, &mut enc_cx);

    // Construct decoder graph
    let mut dec_cx = Graph::new();
    let decoder = model::TextDecoder::initialize(&mut dec_cx);
    let mut text_input = dec_cx.tensor::<(Const<1>, Dyn<'s'>)>();
    let mut encoder_output = (0..model::DEC_LAYERS)
        .map(|_| dec_cx.tensor::<(Const<1>, Dyn<'e'>, Const<{ model::D_MODEL }>)>())
        .collect::<Vec<_>>();
    let mut cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..model::DEC_LAYERS)
        .map(|_| {
            (
                dec_cx.named_tensor("Key Cache"),
                dec_cx.named_tensor("Value Cache"),
            )
        })
        .collect();
    let (mut logits, mut enc_proj_states, mut cache_dest) = decoder.forward((
        &encoder_output,
        text_input,
        &cache_src,
        PhantomData::<Dyn<'t'>>,
    ));
    logits.retrieve();
    enc_proj_states.keep();
    cache_dest.keep();
    loader::load("setup/whisper.gguf", &decoder, &mut dec_cx);

    // Compile graphs
    enc_cx.compile(
        (<(GenericCompiler, MetalCompiler<f32>)>::default()),
        (&mut audio_input, &mut encoded),
    );
    dec_cx.compile(
        (<(GenericCompiler, MetalCompiler<f32>)>::default()),
        (
            &mut text_input,
            &mut encoder_output,
            &mut cache_src,
            &mut logits,
        ),
    );
}
