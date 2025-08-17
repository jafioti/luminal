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
    utils::{binary, build_search_space, loop_in, loop_out, print_kernels},
    GPUArch, GraphTerm,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rustc_hash::FxHashMap;

fn main() {
    autoreleasepool(|| {
        // Make 2.0 graph manually
        let (M, K, N) = (64, 64, 64);
        let mut graph = StableGraph::new();
        let mut a_orig = graph.add_node(GraphTerm::GMEM {
            label: "A".to_string(),
        });
        let mut a = loop_in(
            a_orig,
            M / 8,
            Expression::from('z') * K * 8,
            "m_outer",
            &mut graph,
        );
        a = loop_in(a, N / 8, 0, "n_outer", &mut graph);
        a = loop_in(a, 8, Expression::from('z') * K, "m_tile", &mut graph);
        a = loop_in(a, 8, 0, "n_tile", &mut graph);
        a = loop_in(a, K / 8, Expression::from('z') * 8, "k_outer", &mut graph);
        a = loop_in(a, 8, Expression::from('z'), "k_inner", &mut graph);
        let mut b_orig = graph.add_node(GraphTerm::GMEM {
            label: "B".to_string(),
        });
        let mut b = loop_in(b_orig, M / 8, 0, "m_outer", &mut graph);
        b = loop_in(b, N / 8, Expression::from('z') * 8, "n_outer", &mut graph);
        b = loop_in(b, 8, 0, "m_tile", &mut graph);
        b = loop_in(b, 8, Expression::from('z'), "n_tile", &mut graph);
        b = loop_in(
            b,
            K / 8,
            Expression::from('z') * 8 * N,
            "k_outer",
            &mut graph,
        );
        b = loop_in(b, 8, Expression::from('z') * N, "k_inner", &mut graph);
        let mut acc_orig = graph.add_node(GraphTerm::GMEM {
            label: "acc".to_string(),
        });
        let mut acc = loop_in(acc_orig, M / 8, 0, "m_outer", &mut graph);
        acc = loop_in(acc, N / 8, 0, "n_outer", &mut graph);
        acc = loop_in(acc, 8, 0, "m_tile", &mut graph);
        acc = loop_in(acc, 8, 0, "n_tile", &mut graph);
        acc = loop_in(
            acc,
            K / 8,
            Expression::from(Term::Acc('a')),
            "k_outer",
            &mut graph,
        );
        acc = loop_in(
            acc,
            8,
            Expression::from(Term::Acc('a')),
            "k_inner",
            &mut graph,
        );

        let mut out = binary(
            acc,
            binary(b, a, GraphTerm::Mul, &mut graph),
            GraphTerm::Add,
            &mut graph,
        );

        out = loop_out(
            out,
            8,
            Expression::from(Term::Acc('a')),
            "k_inner",
            &mut graph,
        );
        out = loop_out(
            out,
            8,
            Expression::from(Term::Acc('a')),
            "k_outer",
            &mut graph,
        );
        out = loop_out(out, 8, Expression::from('z'), "n_tile", &mut graph);
        out = loop_out(out, 8, Expression::from('z') * N, "m_tile", &mut graph);
        out = loop_out(out, N / 8, Expression::from('z') * 8, "n_outer", &mut graph);
        loop_out(
            out,
            M / 8,
            Expression::from('z') * 8 * N,
            "m_outer",
            &mut graph,
        );

        let egraph = build_search_space(&graph, 3);
        let inputs = make_test_inputs(&graph, &FxHashMap::default());
        let out_graph = search(
            &egraph,
            &inputs,
            GPUArch::Metal(HashMap::default()),
            &FxHashMap::default(),
        )
        .unwrap();
        a_orig = out_graph
            .node_indices()
            .find(|i| {
                *out_graph.node_weight(*i).unwrap()
                    == GraphTerm::GMEM {
                        label: "A".to_string(),
                    }
            })
            .unwrap();
        b_orig = out_graph
            .node_indices()
            .find(|i| {
                *out_graph.node_weight(*i).unwrap()
                    == GraphTerm::GMEM {
                        label: "B".to_string(),
                    }
            })
            .unwrap();
        acc_orig = out_graph
            .node_indices()
            .find(|i| {
                *out_graph.node_weight(*i).unwrap()
                    == GraphTerm::GMEM {
                        label: "acc".to_string(),
                    }
            })
            .unwrap();
        out = out_graph.externals(Direction::Outgoing).next().unwrap();

        let (kernels, gmem_mapping) = codegen(
            out_graph,
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
        let acc_buffer = copy_metal_buffer(&vec![0.0], &device);

        let mut inputs = FxHashMap::default();
        inputs.insert(gmem_mapping[&a_orig], (a_buffer, false));
        inputs.insert(gmem_mapping[&b_orig], (b_buffer, false));
        inputs.insert(gmem_mapping[&acc_orig], (acc_buffer, false));

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
