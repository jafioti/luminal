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
        // let (M, K, N) = (512, 512, 512);
        // let mut cx = Graph::new();
        // let a = cx.named_tensor("A", (M, K));
        // let b = cx.named_tensor("B", (K, N));
        // let out = a.matmul(b);
        // let (graph, gmem_mapping, accs) = translate_graph_meta(&cx);
        // let graph = graph.node_weights().next().unwrap().clone();
        // luminal_2::utils::display_graph(&graph, &[]);
        // Make 2.0 graph manually
        let (M, K, N) = (512, 512, 512);
        let mut graph = StableGraph::new();
        let mut a_orig = graph.add_node(GraphTerm::GMEM {
            label: "A".to_string(),
        });
        let mut a = loop_in(a_orig, M, Expression::from('z') * K, "m", &mut graph);
        a = loop_in(a, N, 0, "n", &mut graph);
        a = loop_in(a, K, Expression::from('z'), "k", &mut graph);
        let mut b_orig = graph.add_node(GraphTerm::GMEM {
            label: "B".to_string(),
        });
        let mut b = loop_in(b_orig, M, 0, "m", &mut graph);
        b = loop_in(b, N, Expression::from('z'), "n", &mut graph);
        b = loop_in(b, K, Expression::from('z') * N, "k", &mut graph);
        let mut mul_out = binary(a, b, GraphTerm::Mul, &mut graph);
        mul_out = loop_out(mul_out, K, Expression::from('z'), "k", &mut graph);
        mul_out = loop_out(mul_out, N, Expression::from('z') * K, "n", &mut graph);
        mul_out = loop_out(mul_out, M, Expression::from('z') * K * N, "m", &mut graph);
        mul_out = loop_in(mul_out, M, Expression::from('z') * K * N, "m", &mut graph);
        mul_out = loop_in(mul_out, N, Expression::from('z') * K, "n", &mut graph);
        mul_out = loop_in(mul_out, 1, 0, "pad", &mut graph);
        mul_out = loop_in(mul_out, 1, 0, "pad1", &mut graph);
        mul_out = loop_in(mul_out, K, Expression::from('z'), "k", &mut graph);
        let acc_orig = graph.add_node(GraphTerm::GMEM {
            label: "acc".to_string(),
        });
        let mut acc = loop_in(acc_orig, M, 0, "m", &mut graph);
        acc = loop_in(acc, N, 0, "n", &mut graph);
        acc = loop_in(acc, 1, 0, "pad", &mut graph);
        acc = loop_in(acc, 1, 0, "pad1", &mut graph);
        acc = loop_in(acc, K, Expression::from(Term::Acc('a')), "k", &mut graph);
        let mut out = binary(mul_out, acc, GraphTerm::Add, &mut graph);
        out = loop_out(out, K, Expression::from(Term::Acc('a')), "k", &mut graph);
        out = loop_out(out, 1, 0, "pad1", &mut graph);
        out = loop_out(out, 1, 0, "pad", &mut graph);
        out = loop_out(out, N, Expression::from('z'), "n", &mut graph);
        loop_out(out, M, Expression::from('z') * N, "m", &mut graph);
        // luminal_2::utils::display_graph(&graph, &[]);
        let egraph = build_search_space(&graph, 1);
        let inputs = make_test_inputs(
            &graph,
            &FxHashMap::default(),
            &[("acc".to_string(), vec![0.0])],
        );
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
        let acc_orig = out_graph.node_indices().find(|i| {
            *out_graph.node_weight(*i).unwrap()
                == GraphTerm::GMEM {
                    label: "acc".to_string(),
                }
        }); // there may be no acc here anymore
        out = out_graph.externals(Direction::Outgoing).next().unwrap();
        luminal_2::utils::display_graph(&out_graph, &[]);
        println!(
            "Rendered: {}",
            luminal_2::utils::render_egglog(&out_graph, "a").0
        );
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
        if let Some(acc_orig) = acc_orig {
            inputs.insert(gmem_mapping[&acc_orig], (acc_buffer, false));
        }

        let (outputs, _) = run_graph(
            &mut inputs,
            &kernels,
            &FxHashMap::default(),
            &compiled,
            &int_buffers,
            &int_buffer_map,
        );
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
