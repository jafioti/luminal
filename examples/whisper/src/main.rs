#![allow(unused)]
use std::{io::Write, marker::PhantomData};

use itertools::Itertools;
// WIP
use luminal::prelude::*;
use model::KVCache;
use tokenizers::Tokenizer;

mod audio;
mod loader;
mod model;

fn main() {
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    print!("Defining graph");
    std::io::stdout().flush().unwrap();
    let now = std::time::Instant::now();

    // Construct encoder graph
    let mut enc_cx = Graph::new();
    let encoder = model::AudioEncoder::initialize(&mut enc_cx);
    let mut encoder_params = params(&encoder);
    enc_cx.keep_tensors(&encoder_params);
    let mut audio_input = enc_cx.tensor::<(Const<1>, Const<{ model::N_MEL_BINS }>, Dyn<'s'>)>();
    let mut encoded: GraphTensor<(Const<1>, Dyn<'d'>, Const<384>)> = encoder
        .forward((audio_input, PhantomData::<Dyn<'d'>>))
        .keep();
    loader::load("setup/whisper-tiny.safetensors", &encoder, &mut enc_cx);

    // Construct decoder graph
    let mut dec_cx = Graph::new();
    let decoder = model::TextDecoder::initialize(&mut dec_cx);
    let mut decoder_params = params(&decoder);
    dec_cx.keep_tensors(&decoder_params);
    let mut text_input = dec_cx.tensor::<(Const<1>, Dyn<'s'>)>();
    let mut encoder_output =
        dec_cx.named_tensor::<(Const<1>, Dyn<'e'>, Const<{ model::D_MODEL }>)>("Enc Output");
    let mut cache_src: Vec<KVCache<Const<1>, Dyn<'p'>>> = (0..model::DEC_LAYERS)
        .map(|_| (dec_cx.named_tensor("Keys"), dec_cx.named_tensor("Values")))
        .collect();
    cache_src.set_dyn(vec![], &[1, 6, 64, 0]);
    let (logits, _, cache_dest) = decoder.forward((
        encoder_output,
        text_input,
        &cache_src,
        PhantomData::<Dyn<'t'>>,
    ));
    let mut logits = logits
        .slice((.., Expression::from('s') - 1.., ..))
        .retrieve();
    cache_dest.keep();
    loader::load("setup/whisper-tiny.safetensors", &decoder, &mut dec_cx);

    // Compile graphs
    println!("\t\t - {}ms", now.elapsed().as_millis());

    print!("Compiling graph");
    std::io::stdout().flush().unwrap();
    let now = std::time::Instant::now();
    enc_cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::MetalCompiler::<f16>::default(),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f16>::default(),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (&mut audio_input, &mut encoded, &mut encoder_params),
    );
    dec_cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::MetalCompiler::<f16>::default(),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f16>::default(),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (
            &mut text_input,
            &mut encoder_output,
            &mut cache_src,
            &mut logits,
            &mut decoder_params,
        ),
    );
    let cache_src = downstream(cache_src, &dec_cx);
    let encoder_output = downstream(encoder_output, &dec_cx);
    dec_cx.keep_tensors(&encoder_output);
    delete_inputs(&encoder_output, &mut dec_cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    // Load weights
    print!("Loading weights");
    std::io::stdout().flush().unwrap();
    let now = std::time::Instant::now();
    audio_input.set_dyn(vec![0.; 160], &[1, 80, 2]);
    enc_cx.set_dyn_dim('d', 1);
    enc_cx.execute();
    delete_inputs(downstream(encoder_params, &enc_cx), &mut enc_cx);
    text_input.set_dyn(vec![0.], &[1, 1]);
    dec_cx.set_dyn_dim('e', 1);
    dec_cx.set_dyn_dim('p', 0);
    dec_cx.set_dyn_dim('t', 1);
    transfer_data(encoded, &mut enc_cx, &encoder_output, &mut dec_cx);
    dec_cx.execute();
    logits.drop();
    transfer_data_same_graph(&cache_dest, &cache_src, &mut dec_cx);
    delete_inputs(&cache_src, &mut dec_cx);
    delete_inputs(&downstream(decoder_params, &dec_cx), &mut dec_cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    // Process audio into mel spectrogram
    let mel_bytes = include_bytes!("../setup/melfilters.bytes").as_slice();
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    let (pcm_data, _) = audio::pcm_decode("setup/jfk.wav").unwrap();
    let mel = audio::pcm_to_mel(80, &pcm_data, &mel_filters);
    let mel_len = mel.len();

    // Encode audio
    print!("Encoding audio");
    std::io::stdout().flush().unwrap();
    let start_encoding = std::time::Instant::now();

    audio_input.set_dyn(mel, &[1, 80, mel_len / 80]);
    enc_cx.set_dyn_dim('d', (mel_len / 80) / 2);
    enc_cx.execute();
    transfer_data(encoded, &mut enc_cx, encoder_output, &mut dec_cx);
    println!("\t\t - {}ms", start_encoding.elapsed().as_millis());

    // Decode text
    let start_decode = std::time::Instant::now();
    dec_cx.set_dyn_dim('e', mel_len / 80 / 2);
    dec_cx.set_dyn_dim('p', 0);
    dec_cx.set_dyn_dim('t', 3);
    let mut output_ids = vec![];
    text_input.set_dyn(vec![50257., 50358., 50362.], &[1, 3]);
    dec_cx.execute();
    let mut output_token = argmax(&logits.data());
    logits.drop();
    output_ids.push(output_token);
    let output_str = tokenizer.decode(&output_ids, false).unwrap();
    print!("{}", output_str.trim());
    std::io::stdout().flush().unwrap();
    let mut prev_output_len = output_str.len();

    for i in 0..100 {
        transfer_data_same_graph(&cache_dest, &cache_src, &mut dec_cx);
        text_input.set_dyn(vec![output_token as f32], &[1, 1]);
        dec_cx.set_dyn_dim('p', i + 3);
        dec_cx.set_dyn_dim('t', i + 4);
        dec_cx.execute();
        output_token = argmax(&logits.data());
        if output_token == 50256 {
            println!();
            break;
        }
        logits.drop();
        output_ids.push(output_token);
        let output_str = tokenizer.decode(&output_ids, false).unwrap();
        print!("{}", &output_str[prev_output_len..]);
        std::io::stdout().flush().unwrap();
        prev_output_len = output_str.len();
    }
    let avg_token_time =
        start_decode.elapsed().as_micros() as f32 / (output_ids.len() - 3) as f32 / 1000.0;
    println!(
        "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
        avg_token_time,
        1000.0 / avg_token_time
    );
    println!(
        "Total transcription time - {}ms",
        start_encoding.elapsed().as_millis()
    );
}

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
