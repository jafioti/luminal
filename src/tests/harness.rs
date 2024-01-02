use petgraph::stable_graph::NodeIndex;

use crate::prelude::*;

use super::{assert_close, assert_exact};

// #[allow(clippy::type_complexity)]
// pub fn test_compilers_exact(
//     graphs: &[fn() -> (Graph, Vec<GraphTensor<()>>)],
//     compilers: &[Box<dyn Compiler>],
// ) {
//     test_compilers(graphs, compilers, assert_exact)
// }

// #[allow(clippy::type_complexity)]
// pub fn test_compilers_close(
//     graphs: &[fn() -> (Graph, Vec<GraphTensor<()>>)],
//     compilers: &[Box<dyn Compiler>],
// ) {
//     test_compilers(graphs, compilers, assert_close)
// }

// #[allow(clippy::type_complexity)]
// fn test_compilers<F: Fn(&[f32], &[f32])>(
//     graphs: &[fn() -> (Graph, Vec<GraphTensor<()>>)],
//     compilers: &[Box<dyn Compiler>],
//     condition: F,
// ) {
//     for create_graph in graphs {
//         let (mut cx, result_tensors) = create_graph();
//         cx.execute();
//         let unopt_results = result_tensors
//             .into_iter()
//             .map(|mut gt| {
//                 gt.graph_ref = &mut cx;
//                 gt.data()
//             })
//             .collect::<Vec<_>>();

//         for compiler in compilers {
//             let (mut cx, result_tensors) = create_graph();
//             compiler.compile(&mut cx);
//             cx.execute();
//             let opt_results = result_tensors
//                 .into_iter()
//                 .map(|mut gt| {
//                     gt.graph_ref = &mut cx;
//                     gt.data()
//                 })
//                 .collect::<Vec<_>>();
//             for (a, b) in unopt_results.iter().zip(opt_results.iter()) {
//                 condition(a, b);
//             }
//         }
//     }
// }
