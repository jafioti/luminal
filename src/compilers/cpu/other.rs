use crate::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
    shape::symbolic::BigExpression,
};
use rustc_hash::FxHashMap;

#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct ARange {
    pub size: BigExpression,
    dyn_map: *const FxHashMap<char, usize>,
}

impl Operator for ARange {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let n_elements = self
            .size
            .exec(unsafe { self.dyn_map.as_ref().unwrap() })
            .unwrap();
        vec![Tensor {
            data: Box::new((0..n_elements).map(|i| i as f32).collect::<Vec<_>>()),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct ARangeCompiler;

impl Compiler for ARangeCompiler {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let (
            mut one_const,
            mut contig1,
            mut contig2,
            mut contig3,
            mut contig4,
            mut sum_reduce,
            mut subtraction_constant,
            mut subtraction,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig = SelectOp::new().ty::<Contiguous>();
        let pre_sub_pattern = SelectOp::new()
            .check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<Constant>() {
                    match c.0 {
                        ConstantValue::Float(f) => f == 1.0,
                        _ => false,
                    }
                } else {
                    false
                }
            })
            .ptr(&mut one_const)
            .edge(contig.clone().ptr(&mut contig1))
            .edge(contig.clone().ptr(&mut contig2))
            .edge(contig.clone().ptr(&mut contig3))
            .edge(contig.clone().ptr(&mut contig4))
            .edge(SelectOp::new().ty::<SumReduce>().ptr(&mut sum_reduce));
        let mut s1 = pre_sub_pattern
            .clone()
            .edge(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(c) = o.as_any().downcast_ref::<Constant>() {
                            match c.0 {
                                ConstantValue::Float(f) => f == 1.0,
                                _ => false,
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut subtraction_constant)
                    .edge(
                        SelectOp::new()
                            .ty::<super::binary::Sub>()
                            .ptr(&mut subtraction),
                    ),
            )
            .search(graph);
        let mut s2 = pre_sub_pattern
            .edge(
                SelectOp::new()
                    .check(|o, _| {
                        if let Some(c) = o.as_any().downcast_ref::<Constant>() {
                            match c.0 {
                                ConstantValue::Float(f) => f == -1.0,
                                _ => false,
                            }
                        } else {
                            false
                        }
                    })
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<Add>().ptr(&mut subtraction)),
            )
            .search(graph);

        while s1.next_match() || s2.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(one_const, contig1)
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
                    size: arange_amount.into(),
                    dyn_map: &graph.dyn_map,
                })
                .finish();
            move_outgoing_edge(subtraction, arange_op, &mut graph.graph);

            graph.graph.remove_node(subtraction);
            graph.safe_remove_node(subtraction_constant, 0);
            graph.safe_remove_node(sum_reduce, 0);
            graph.safe_remove_node(contig4, 0);
            graph.safe_remove_node(contig3, 0);
            graph.safe_remove_node(contig2, 0);
            graph.safe_remove_node(contig1, 0);
            graph.safe_remove_node(one_const, 0);
            s1.clear_cached_results();
            s2.clear_cached_results();
        }
    }
}
