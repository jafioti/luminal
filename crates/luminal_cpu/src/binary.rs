use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
};

use super::other::ARange;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sub;

impl Operator for Sub {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (get_vec(&tensors[0].0), get_vec(&tensors[1].0));
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
            data[i] = lhs - rhs;
        }
        vec![Tensor::new(data)]
    }
}

#[derive(Debug, Default)]
pub struct SubtractionCompiler;

impl Compiler for SubtractionCompiler {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let (lhs, rhs) = (node(), node());
        let mul = binary::<Mul>(rhs.clone(), super::constant(-1.));
        let add = binary::<Add>(lhs.clone(), mul.clone());
        let mut s = add.clone().search(graph);

        while s.next_match() {
            if s.check_no_delete(&[add.id]) {
                s.clear_cached_results();
                continue;
            }
            let add = s.get(&add);
            let (a, a_edge) = graph
                .graph
                .edges_connecting(s.get(&lhs), add)
                .next()
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (b, b_edge) = graph
                .graph
                .edges_connecting(s.get(&rhs), s.get(&mul))
                .next()
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let b_final_shape = graph
                .graph
                .edges_connecting(s.get(&mul), add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            if b_final_shape.is_reshaped() {
                continue;
            }
            let sub = graph
                .add_op(Sub)
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(add, sub, &mut graph.graph);

            graph.graph.remove_node(add);
            s.try_delete();
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Equal;

impl Operator for Equal {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_data, b_data) = (get_vec(&tensors[0].0), get_vec(&tensors[1].0));
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
            data[i] = if (a - b).abs() < 1e-6 { 1. } else { 0. };
        }
        vec![Tensor::new(data)]
    }
}

#[derive(Debug, Default)]
pub struct EqualCompiler;

impl Compiler for EqualCompiler {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let one = super::constant(1.);
        let (lhs, rhs) = (node(), node());
        let lt1 = binary::<LessThan>(lhs.clone(), rhs.clone());
        let ne = binary::<Add>(lt1.clone(), binary::<LessThan>(rhs.clone(), lhs.clone()));
        let eq = binary::<Sub>(one, ne);

        let mut s = eq.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[eq.id]) {
                continue;
            }
            let (lhs, rhs) = (s.get(&lhs), s.get(&rhs));
            let eq = s.get(&eq);
            let a_edge = graph
                .graph
                .edges_connecting(lhs, s.get(&lt1))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let b_edge = graph
                .graph
                .edges_connecting(rhs, s.get(&lt1))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let equals = graph
                .add_op(Equal)
                .input(lhs, a_edge.1, a_edge.2)
                .input(rhs, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(eq, equals, &mut graph.graph);

            graph.graph.remove_node(eq);
            s.try_delete();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gather {
    pub embed_dim: usize,
}

impl Operator for Gather {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Inp 1 should be Vec<f32> and inp 2 should be a CudaSlice<T>
        let indexes = tensors[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        let weights = tensors[1].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();

        let mut out = vec![0.; indexes.len() * self.embed_dim];
        for token in 0..indexes.len() {
            let e = indexes[token] as usize;
            for dim in 0..self.embed_dim {
                out[token * self.embed_dim + dim] = weights[e * self.embed_dim + dim];
            }
        }

        vec![Tensor::new(out)]
    }
}

#[derive(Debug, Default)]
pub struct GatherCompiler;

impl Compiler for GatherCompiler {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let indexes = node();
        let eq = binary::<Equal>(indexes.clone(), op::<ARange>());
        let embedding = node();
        let mul = binary::<Mul>(embedding.clone(), eq.clone());
        let sum_reduce = unary::<SumReduce>(mul.clone());
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[embedding.id]) {
                continue;
            }
            let emb_shape = graph
                .edges_connecting(s.get(&embedding), s.get(&mul))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let index_shape = graph
                .edges_connecting(s.get(&indexes), s.get(&eq))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let embed_dim = graph
                .graph
                .edges_connecting(s.get(&embedding), s.get(&mul))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2
                .dims()[2]
                .to_usize()
                .unwrap();

            let gather = graph
                .add_op(Gather { embed_dim })
                .input(s.get(&indexes), 0, index_shape)
                .input(s.get(&embedding), 0, emb_shape)
                .finish();
            move_outgoing_edge(s.get(&sum_reduce), gather, &mut graph.graph);
            graph.remove_node(s.get(&sum_reduce));
            s.try_delete();
        }
    }
}

fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap()
}
