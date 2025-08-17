use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::{
    petgraph::{prelude::StableGraph, visit::EdgeRef, Direction},
    *,
};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::{binary, loop_in, loop_out, print_kernels},
    GPUArch, GraphTerm,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rustc_hash::FxHashMap;

fn main() {
    autoreleasepool(|| {
        // Make 2.0 graph manually
        let (M, K, N) = (64, 64, 64);
        let mut graph = StableGraph::new();
        let a_orig = graph.add_node(GraphTerm::GMEM {
            label: "A".to_string(),
        });
        let mut a = loop_in(a_orig, M / 8, Expression::from('z') * K * 8, "", &mut graph);
        a = loop_in(a, N / 8, 0, "", &mut graph);
        a = loop_in(a, 8, 0, "", &mut graph);
        a = loop_in(a, 4, 0, "", &mut graph);
        let b_orig = graph.add_node(GraphTerm::GMEM {
            label: "B".to_string(),
        });
        let mut b = loop_in(b_orig, M / 8, 0, "", &mut graph);
        b = loop_in(b, N / 8, Expression::from('z') * 8, "", &mut graph);
        b = loop_in(b, 8, 0, "", &mut graph);
        b = loop_in(b, 4, 0, "", &mut graph);
        let mut out = binary(
            a,
            b,
            GraphTerm::TCMatmul {
                a_k_stride: Expression::from('z') * 8,
                b_k_stride: Expression::from('z') * N * 8,
                a_inner_stride: K.into(),
                b_inner_stride: N.into(),
                c_inner_stride: N.into(),
                k_outer_loops: (K / 8).into(),
            },
            &mut graph,
        );
        out = loop_out(out, 4, 0, "", &mut graph);
        out = loop_out(out, 8, 0, "", &mut graph);
        out = loop_out(out, N / 8, Expression::from('z') * 8, "", &mut graph);
        out = loop_out(out, M / 8, Expression::from('z') * 8 * N, "", &mut graph);

        let (kernels, gmem_mapping) = codegen(
            graph,
            vec![out],
            GPUArch::Metal(HashMap::default()),
            0,
            &HashMap::default(),
            false,
        )
        .unwrap();

        let compiled = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);

        let device = Device::system_default().unwrap();
        let a_buffer = copy_metal_buffer(&vec![1.; M * K], &device);
        let b_buffer = copy_metal_buffer(&vec![1.; K * M], &device);

        let mut inputs = FxHashMap::default();
        inputs.insert(gmem_mapping[&a_orig], (a_buffer, false));
        inputs.insert(gmem_mapping[&b_orig], (b_buffer, false));

        let (outputs, _) = run_graph(
            &mut inputs,
            &kernels,
            &FxHashMap::default(),
            &compiled,
            &int_buffers,
            &int_buffer_map,
        );
        println!("{}", print_kernels(&kernels));
        println!("{:?}", &copy_metal_buffer_back(&outputs[0])[..10]);
    });
    expression_cleanup();
}

pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    let buf = device.new_buffer_with_data(
        v.as_ptr() as *const _,
        (v.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf
}

pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}
