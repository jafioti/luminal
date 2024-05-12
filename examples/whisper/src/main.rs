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
    enc_cx.keep_tensors(params(&encoder));
    let mut audio_input = enc_cx.tensor::<(Const<1>, Dyn<'s'>, Const<{ model::N_MEL_BINS }>)>();
    let mut encoded = encoder.forward(audio_input);
    encoded.retrieve();
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
    cache_src.set_dyn(vec![], &[1, 6, 0, 64]);
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

    // Process audio into mel spectrogram
    let mel_bytes = include_bytes!("../setup/melfilters.bytes").as_slice();
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    let (pcm_data, sample_rate) = audio::pcm_decode("setup/samples_jfk.wav").unwrap();
    let mel = audio::pcm_to_mel(80, &pcm_data, &mel_filters);
    let mel_len = mel.len();

    // Encode audio
    audio_input.set_dyn(mel, &[1, mel_len / 80, 80]);
    enc_cx.execute();
    transfer_data(encoded, &mut enc_cx, encoder_output, &mut dec_cx);

    // // Decode text
    // for _ in 0..1000 {

    // }
}
