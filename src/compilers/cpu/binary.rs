use crate::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
};
use itertools::Itertools;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sub;

impl Operator for Sub {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&tensors[0].0),
            get_vec_from_tensor(&tensors[1].0),
        );
        let (a_ind, a_val, b_ind, b_val) = (
            tensors[0].1.index_expression(),
            tensors[0].1.valid_expression(),
            tensors[1].1.index_expression(),
            tensors[1].1.valid_expression(),
        );
        let mut data = vec![0.; tensors[0].1.n_elements().to_usize().unwrap()];
        for i in 0..data.len() {
            let lhs = if a_val.exec_single_var(i) != 0 {
                a_data[a_ind.exec_single_var(i)]
            } else {
                0.0
            };
            let rhs = if b_val.exec_single_var(i) != 0 {
                b_data[b_ind.exec_single_var(i)]
            } else {
                0.0
            };
            data[i] = lhs + rhs;
        }
        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct SubtractionCompiler;

impl Compiler for SubtractionCompiler {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let (mut neg_one, mut mul, mut add) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let mut searcher = SelectOp::new()
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
            .ptr(&mut neg_one)
            .edge(SelectOp::new().ty::<Mul>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<Add>().ptr(&mut add))
            .search(graph);

        while searcher.next_match() {
            if check_no_delete(graph, &[neg_one, mul, add]) {
                continue;
            }
            let (a, a_edge) = graph
                .graph
                .edges_directed(add, petgraph::Direction::Incoming)
                .find(|e| e.source() != mul)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (b, b_edge) = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != neg_one)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let b_final_shape = graph
                .graph
                .edges_connecting(mul, add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            if !b_final_shape.is_contiguous()
                || b_final_shape.is_sliced()
                || b_final_shape.is_padded()
            {
                continue;
            }
            let sub = graph
                .add_op(Sub)
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(add, sub, &mut graph.graph);

            if graph.get_dests(neg_one).len() == 1 {
                graph.graph.remove_node(neg_one);
            }
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Equal;

impl Operator for Equal {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (
            get_vec_from_tensor(&tensors[0].0),
            get_vec_from_tensor(&tensors[1].0),
        );
        let mut data = vec![0.; tensors[0].1.n_elements().to_usize().unwrap()];
        let (a_ind, a_val, b_ind, b_val) = (
            tensors[0].1.index_expression(),
            tensors[0].1.valid_expression(),
            tensors[1].1.index_expression(),
            tensors[1].1.valid_expression(),
        );
        for i in 0..data.len() {
            let a = if a_val.exec_single_var(i) != 0 {
                a_data[a_ind.exec_single_var(i)]
            } else {
                0.0
            };
            let b = if b_val.exec_single_var(i) != 0 {
                b_data[b_ind.exec_single_var(i)]
            } else {
                0.0
            };
            data[i] = if a < b { 1. } else { 0. };
        }
        vec![Tensor {
            data: Box::new(data),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct EqualCompiler;

impl Compiler for EqualCompiler {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let (mut less_than1, mut less_than2, mut add, mut one, mut sub) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
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
            .ptr(&mut one)
            .edge(
                SelectOp::new()
                    .ty::<LessThan>()
                    .ptr(&mut less_than1)
                    .edge(
                        SelectOp::new()
                            .ty::<LessThan>()
                            .ptr(&mut less_than2)
                            .edge(SelectOp::new().ty::<Add>().ptr(&mut add)),
                    )
                    .edge(SelectOp::new().ty::<Sub>().ptr(&mut sub)),
            );

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            let lt1_inputs = graph
                .graph
                .neighbors_directed(less_than1, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            let lt2_inputs = graph
                .graph
                .neighbors_directed(less_than2, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            if lt1_inputs != lt2_inputs {
                continue;
            }
            let inputs = graph
                .graph
                .edges_directed(less_than1, petgraph::Direction::Incoming)
                .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                .map(|e| e.source())
                .collect::<Vec<_>>();
            let (a, b) = (inputs[0], inputs[1]);
            if check_no_delete(graph, &[less_than1, less_than2, add, one, sub]) {
                continue;
            }
            let a_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(a, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let b_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(b, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let equals = graph
                .add_op(Equal)
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(sub, equals, &mut graph.graph);

            graph.graph.remove_node(sub);
            graph.safe_remove_node(add, 0);
            graph.safe_remove_node(one, 0);
            graph.safe_remove_node(less_than2, 0);
            graph.safe_remove_node(less_than1, 0);
            searcher.clear_cached_results();
        }
    }
}

#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct Gather {
    pub embed_dim: usize,
}

impl Operator for Gather {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Inp 1 should be Vec<f32> and inp 2 should be a CudaSlice<T>
        let indexes = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let weights = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();

        let mut out = vec![0.; indexes.len() * self.embed_dim];
        for token in 0..indexes.len() {
            let e = indexes[token] as usize;
            for dim in 0..self.embed_dim {
                out[token * self.embed_dim + dim] = weights[e * self.embed_dim + dim];
            }
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct GatherCompiler;

impl Compiler for GatherCompiler {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let (mut arange, mut equal, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
            .ty::<super::other::ARange>()
            .ptr(&mut arange)
            .edge(SelectOp::new().ty::<Equal>().ptr(&mut equal))
            .edge(SelectOp::new().ty::<Mul>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<SumReduce>().ptr(&mut sum_reduce));
        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[arange, equal, mul, sum_reduce]) {
                continue;
            }
            let embed_dim = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != equal && !e.weight().is_schedule())
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2
                .shape()[2]
                .to_usize()
                .unwrap();
            let gather = graph.add_op(Gather { embed_dim }).finish();
            move_incoming_edge(equal, gather, &mut graph.graph);
            graph.safe_remove_node(equal, 1);
            move_incoming_edge(mul, gather, &mut graph.graph);
            move_outgoing_edge(sum_reduce, gather, &mut graph.graph);
            graph.graph.remove_node(sum_reduce);
            graph.safe_remove_node(mul, 0);
            graph.safe_remove_node(arange, 0);
        }
    }
}
