mod config;
mod loader;
mod model;

use std::collections::HashMap;

use cudarc::{
    driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchConfig},
    nvrtc::compile_ptx,
};
use luminal::{
    nn::{activation::RMSNorm, transformer::encoder::TransformerEncoder},
    op::{InputTensor, Operator},
    prelude::{prim::CudaMul, *},
};
use model::LlamaForCausalLM;
use rand::Rng;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

use crate::model::Llama;

#[rustfmt::skip]
fn main() {
    // let dev = CudaDevice::new(0).unwrap();
    // let op = CudaMul::new(ShapeTracker::new(&[Dim::Known(1), Dim::Known(10), Dim::Known(4096)]), ShapeTracker::new(&[Dim::Known(1), Dim::Known(10), Dim::Known(4096)]), dev.clone(), &mut HashMap::new());
    // let mut rng = rand::thread_rng();
    // let cpu_data = (0..40960)
    //     .map(|_| rng.gen_range(-0.01..0.01))
    //     .collect::<Vec<f32>>();
    // println!("In: {:?}", &cpu_data[cpu_data.len().saturating_sub(10)..]);
    // let mut a: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
    // dev.htod_sync_copy_into(&cpu_data, &mut a).unwrap();
    // let mut b: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
    // dev.htod_sync_copy_into(&cpu_data, &mut b).unwrap();
    // let out = op.process(vec![(InputTensor::Owned(Tensor{ 
    //     data: Box::new(a),
    // }), ShapeTracker::new(&[Dim::Known(1), Dim::Known(10), Dim::Known(4096)])), (InputTensor::Owned(Tensor{ 
    //     data: Box::new(b),
    // }), ShapeTracker::new(&[Dim::Known(1), Dim::Known(10), Dim::Known(4096)]))]);
    // let o = out[0]
    //         .data
    //         .as_any()
    //         .downcast_ref::<CudaSlice<f32>>()
    //         .unwrap();
    // let o = dev.dtoh_sync_copy(o).unwrap();
    // println!("Out: {:?}", &o[o.len().saturating_sub(10)..]);




//     let code = "extern \"C\" __global__ void kernel_mul(float *out, const float *inp_a, const float *inp_b, int numel) {{
// int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numel) {{
//         out[idx] = idx;
//     }}
// }}";
    // dev.load_ptx(
    //     compile_ptx(code).unwrap(),
    //     "kernel_mul",
    //     &["kernel_mul"],
    // )
    // .unwrap();

    // let out = unsafe { dev.alloc::<f32>(40960) }.unwrap();
    // let cpu_data = vec![0.0; 40960];
    // let mut a: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
    // dev.htod_sync_copy_into(&cpu_data, &mut a).unwrap();
    // let mut b: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
    // dev.htod_sync_copy_into(&cpu_data, &mut b).unwrap();
    // let mut params = vec![
    //     (&out).as_kernel_param(),
    //     (&a).as_kernel_param(),
    //     (&b).as_kernel_param(),
    //     40960.as_kernel_param(),
    // ];
    // let f = dev.get_func("kernel_mul", "kernel_mul")
    // .unwrap();
    // unsafe {
    //     f
    //         .clone()
    //         .launch_async_impl(LaunchConfig::for_num_elems(40960), &mut params)
    //         .unwrap();
    // }
    // let a = dev.dtoh_sync_copy(&a).unwrap();
    // println!("A: {:?}", &a[a.len().saturating_sub(10)..]);
    // let b = dev.dtoh_sync_copy(&b).unwrap();
    // println!("B: {:?}", &b[b.len().saturating_sub(10)..]);
    // let o = dev.dtoh_sync_copy(&out).unwrap();
    // println!("Out: {:?}", &o[o.len().saturating_sub(10)..]);


    // let mut cx = Graph::new();
    // let a = cx.new_tensor::<(Const<1>, Const<2>)>("Input");
    // let model = LlamaForCausalLM::<{ config::VOCAB },
    //     { config::HEADS },
    //     { config::HIDDEN },
    //     { config::INTERMEDIATE },
    //     { config::HEAD_DIM },
    //     { config::HEAD_DIM_OVER_2 },
    //     { config::LAYERS },>::initialize(&mut cx);
    // // let mut rng = rand::thread_rng();
    // // let data = (0..40960)
    // //     .map(|_| rng.gen_range(-0.01..0.01))
    // //     .collect::<Vec<f32>>();
    // a.set_dyn(vec![1, 2], vec![1, 2]);
    // let (b, _) = model.forward(a);
    // loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx);
    // b.mark();

    // cx.optimize(<(CudaOptimizer, GenericOptimizer)>::default());
    // cx.execute_debug();

    let prompt = "Here is a python implementation of merge sort:";
    let tokenizer =
            SentencePieceBpeTokenizer::from_file("./examples/llama/setup/llama-7b-hf/tokenizer.model", false).unwrap();
    let mut input: Vec<usize> = tokenizer.encode(
        prompt,
        None,
        prompt.len(),
        &TruncationStrategy::LongestFirst,
        0
    ).token_ids.iter() .map(|&x| x as usize).collect();
    input.insert(0, 1);

    println!("Creating Graphs...");
    let mut cx1 = Graph::new();
    // let mut cx2 = Graph::new();
    let mut model: LlamaForCausalLM<
        { config::VOCAB },
        { config::HEADS },
        { config::HIDDEN },
        { config::INTERMEDIATE },
        { config::HEAD_DIM },
        { config::HEAD_DIM_OVER_2 },
        { config::LAYERS },
    > = InitModule::initialize(&mut cx1);
    let inp = cx1.new_tensor::<(Dyn<'b'>, Const<10>)>("Input");
    let (out1, cache1) = model.forward(inp);
    out1.mark();
    for (k, v) in &cache1 {
        k.mark_no_delete();
        v.mark_no_delete();
    }
    loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx1);
    // cx1.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
    cx1.optimize(<(CudaOptimizer, GenericOptimizer)>::default());

    // // Build KV cache forward graph
    // // model = InitModule::initialize(&mut cx2);
    // // let single_inp = cx2.new_tensor::<(Dyn<'b'>, Const<1>)>("Input");
    // // let cache_src = (0..config::LAYERS).map(|_| (cx2.new_tensor("Key Cache"), cx2.new_tensor("Value Cache"))).collect::<Vec<_>>();
    // // let (out, cache_dest)= model.forward_kv::<_, _, Dyn<'s'>, Dyn<'t'>>((single_inp, cache_src.clone()));
    // // out.mark();
    // // for (k, v) in &cache_dest {
    // //     k.mark_no_delete();
    // //     v.mark_no_delete();
    // // }
    // // loader::DfdxDeferredLoader::new("./examples/llama/setup/llama-7b-hf").load(&model, &mut cx2);
    // // cx2.optimize(<(CudaOptimizer, GenericOptimizer)>::default());

    println!("Inferencing...");
    // First pass
    inp.set_dyn(input.clone(), vec![1, 10]);
    // cx1.display_shapes();
    cx1.execute_debug();

    let out1 = out1.dyn_data(&cx1.dyn_map);
    println!("{:?}", &out1[out1.len() - 10..]);
    input.push(sample_index(&out1[out1.len() - 32_000..]));
    println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false));

    // Move cache over to second graph
    // for ((key_src, val_src), (key_dest, val_dest)) in cache1.into_iter().zip(cache_src.iter()) {
    //     cx2.set_tensor(key_dest.id, 0, cx1.get_tensor(key_src.id, 0).unwrap());
    //     cx2.set_tensor(val_dest.id, 0, cx1.get_tensor(val_src.id, 0).unwrap());
    //     key_dest.mark_no_delete();
    //     val_dest.mark_no_delete();
    // }
    // drop(cx1);

    // loop {
    //     single_inp.set_dyn(vec![*input.last().unwrap()], vec![1, 1]);
    //     cx2.set_dyn_dim('s', input.len() - 1);
    //     cx2.set_dyn_dim('t', input.len());

    //     let now = std::time::Instant::now();
    //     cx2.execute();
    //     println!("Forward Pass Took {:.2}s", now.elapsed().as_secs_f32());
        
    //     let o = out.dyn_data(&cx2.dyn_map);
    //     out.drop();
    //     // Sample tokens
    //     input.push(sample_index(&o));
    //     println!("{}", tokenizer.decode(&input.iter().map(|i| *i as i64).collect::<Vec<_>>(), true, false).replace("<0x0A>", "\n"));

    //     // Swap caches
    //     for ((src_k, src_v), (dest_k, dest_v)) in cache_src.iter().copied().zip(cache_dest.iter().copied()) {
    //         // Move dest caches to src
    //         cx2.swap_tensors(src_k, dest_k);
    //         cx2.swap_tensors(src_v, dest_v);
    //         // Drop dest caches
    //         dest_k.drop();
    //         dest_v.drop();
    //     }
    // }
}

// Currently just an argmax, do actual sampling here
fn sample_index(dist: &[f32]) -> usize {
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}
