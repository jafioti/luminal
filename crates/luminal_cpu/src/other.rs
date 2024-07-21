use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
};
use rustc_hash::FxHashMap;

use super::binary::Sub;

#[derive(Debug, Clone, PartialEq)]
pub struct ARange {
    pub size: Expression,
    dyn_map: *const FxHashMap<char, usize>,
}

impl Operator for ARange {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let n_elements = self
            .size
            .exec(unsafe { self.dyn_map.as_ref().unwrap() })
            .unwrap();
        vec![Tensor::new(
            (0..n_elements).map(|i| i as f32).collect::<Vec<_>>(),
        )]
    }
}

#[derive(Debug, Default)]
pub struct ARangeCompiler;

impl Compiler for ARangeCompiler {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let one1 = super::constant(1.);
        let one2 = super::constant(1.);
        let contig1 = unary::<Contiguous>(one1.clone());
        let sum_reduce =
            unary::<SumReduce>(unary::<Contiguous>(unary::<Contiguous>(
                unary::<Contiguous>(contig1.clone()),
            )));
        let sub = binary::<Sub>(sum_reduce, one2.clone());
        let mut s = sub.clone().search(graph);

        while s.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(s.get(&one1), s.get(&contig1))
                            .next()
                            .unwrap()
                            .id(),
                    )
                    .unwrap()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(ARange {
                    size: arange_amount,
                    dyn_map: &graph.dyn_map,
                })
                .finish();
            move_outgoing_edge(s.get(&sub), arange_op, &mut graph.graph);
            graph.graph.remove_node(s.get(&sub));
            s.try_delete();
        }
    }
}
