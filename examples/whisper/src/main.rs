#![allow(unused)]
use std::{io::Write, marker::PhantomData};

use itertools::Itertools;
// WIP
use luminal::prelude::*;
use luminal_metal::MetalCompiler;
use model::KVCache;
use tokenizers::Tokenizer;

mod audio;
mod gguf;
mod loader;
mod model;

fn main() {
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    // Construct encoder graph
    let mut enc_cx = Graph::new();
    let encoder = model::AudioEncoder::initialize(&mut enc_cx);
    enc_cx.keep_tensors(params(&encoder));
    let mut audio_input = enc_cx.tensor::<(Const<1>, Dyn<'s'>, Const<{ model::N_MEL_BINS }>)>();
    let mut encoded = encoder.forward(audio_input);
    encoded.keep();
    loader::load("setup/whisper.gguf", &encoder, &mut enc_cx);

    // Construct decoder graph
    let mut dec_cx = Graph::new();
    let decoder = model::TextDecoder::initialize(&mut dec_cx);
    dec_cx.keep_tensors(params(&decoder));
    let mut text_input = dec_cx.tensor::<(Const<1>, Dyn<'s'>)>();
    let mut encoder_output = (0..model::DEC_LAYERS)
        .map(|_| {
            dec_cx.named_tensor::<(Const<1>, Dyn<'e'>, Const<{ model::D_MODEL }>)>("Enc Output")
        })
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
    let (logits, _, mut cache_dest) = decoder.forward((
        &encoder_output,
        text_input,
        &cache_src,
        PhantomData::<Dyn<'t'>>,
    ));
    let mut logits = logits
        .slice((.., Expression::from('s') - 1.., ..))
        .retrieve();
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
    let cache_src = downstream(cache_src, &dec_cx);
    let encoder_output = downstream(&encoder_output, &dec_cx);
    dec_cx.keep_tensors(&encoder_output);
    delete_inputs(&encoder_output, &mut dec_cx);

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

    // Decode text
    dec_cx.set_dyn_dim('e', enc_cx.dyn_map[&'s']);
    dec_cx.set_dyn_dim('t', 1);
    let mut output_tokens = vec![];
    text_input.set_dyn(vec![0.0], &[1, 1]);
    dec_cx.execute();
    delete_inputs(&cache_src, &mut dec_cx);
    let output_token = argmax(&logits.data());
    output_tokens.push(output_token);
    let output_str = tokenizer.decode(&output_tokens, false).unwrap();
    print!("{output_str}");
    std::io::stdout().flush().unwrap();
    let mut prev_output_len = output_str.len();

    for i in 1..100 {
        transfer_data_same_graph(&cache_dest, &cache_src, &mut dec_cx);
        text_input.set_dyn(vec![output_token as f32], &[1, 1]);
        dec_cx.set_dyn_dim('p', i);
        dec_cx.set_dyn_dim('t', i + 1);
        dec_cx.execute();
        let output_token = argmax(&logits.data());
        logits.drop();
        output_tokens.push(output_token);
        let output_str = tokenizer.decode(&output_tokens, false).unwrap();
        print!("{}", &output_str[prev_output_len..]);
        std::io::stdout().flush().unwrap();
        prev_output_len = output_str.len();
    }
}

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
