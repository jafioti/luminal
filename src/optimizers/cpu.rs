use std::any::Any;

use itertools::Itertools;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{
    op::{Exp2, InputTensor, Log2, Mul, Operator, Recip, Sin, Sqrt, SumReduce},
    prelude::*,
};

// Ops and optimizers specific to CPU execution

pub type CPUOptimizer = (MatMulOptimizer, UnaryFusionOptimizer);

pub type MatMulOptimizer = (MatMul2DOptimizer, BatchMatMul2DOptimizer);

#[derive(Debug, Default)]
pub struct MatMul2DOptimizer;

impl GraphOptimizer for MatMul2DOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // Look for the matmul pattern
        let s = GraphSelector::default();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        s.edge(
            s.op()
                .ty::<Mul>()
                .shapes(vec![
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                    vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
                ])
                .fakes(vec![vec![false, true, false], vec![true, false, false]])
                .ptr(&mut mul),
            s.op()
                .ty::<SumReduce>()
                .value(SumReduce(2))
                .ptr(&mut sum_reduce),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].1.remove_dim(1);
            srcs[1].1.remove_dim(0);
            srcs[1].1.permute(&[1, 0]);
            let new_op = graph
                .add_op(MatMul2D)
                .input(srcs[0].0, 0, srcs[0].1)
                .input(srcs[1].0, 0, srcs[1].1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MatMul2D;

impl Operator for MatMul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let b_data = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut c = vec![0.; a_shape[0].to_usize().unwrap() * b_shape[1].to_usize().unwrap()];
        unsafe {
            matrixmultiply::sgemm(
                a_shape[0].to_usize().unwrap(),
                a_shape[1].to_usize().unwrap(),
                b_shape[1].to_usize().unwrap(),
                1.0,
                &a_data[0],
                a_strides[0] as isize,
                a_strides[1] as isize,
                &b_data[0],
                b_strides[0] as isize,
                b_strides[1] as isize,
                0.0,
                &mut c[0],
                b_shape[1].to_usize().unwrap() as isize,
                1,
            );
        }

        vec![Tensor { data: Box::new(c) }]
    }
}

#[derive(Debug, Default)]
pub struct BatchMatMul2DOptimizer;

impl GraphOptimizer for BatchMatMul2DOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // Look for the matmul pattern
        let s = GraphSelector::default();
        let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        s.edge(
            s.op()
                .ty::<Mul>()
                .shapes(vec![
                    vec![
                        Dim::Unknown('Z'),
                        Dim::Unknown('A'),
                        Dim::Unknown('C'),
                        Dim::Unknown('B'),
                    ],
                    vec![
                        Dim::Unknown('Z'),
                        Dim::Unknown('A'),
                        Dim::Unknown('C'),
                        Dim::Unknown('B'),
                    ],
                ])
                .fakes(vec![
                    vec![false, false, true, false],
                    vec![true, true, false, false],
                ])
                .ptr(&mut mul),
            s.op()
                .ty::<SumReduce>()
                .value(SumReduce(3))
                .ptr(&mut sum_reduce),
        );
        for _ in s.search(graph) {
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert MatMul2D op
            let mut srcs = graph.get_sources(mul);
            // Undo expansions and permute
            srcs[0].1.remove_dim(2);
            srcs[1].1.remove_dim(1);
            srcs[1].1.remove_dim(0);
            srcs[1].1.permute(&[1, 0]);
            let new_op = graph
                .add_op(BatchedMatMul2D)
                .input(srcs[0].0, 0, srcs[0].1)
                .input(srcs[1].0, 0, srcs[1].1)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                sum_reduce,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BatchedMatMul2D;

// ABCxCD -> ABD
impl Operator for BatchedMatMul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let a_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let b_data = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut c = vec![
            0.;
            a_shape[0].to_usize().unwrap()
                * a_shape[1].to_usize().unwrap()
                * b_shape[1].to_usize().unwrap()
        ];

        let logical_batch_size = a_shape
            .iter()
            .skip(1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>();
        for i in 0..a_shape[0].to_usize().unwrap() {
            unsafe {
                matrixmultiply::sgemm(
                    a_shape[1].to_usize().unwrap(),
                    a_shape[2].to_usize().unwrap(),
                    b_shape[1].to_usize().unwrap(),
                    1.0,
                    &a_data[i * a_strides[0]],
                    a_strides[1] as isize,
                    a_strides[2] as isize,
                    &b_data[0],
                    b_strides[0] as isize,
                    b_strides[1] as isize,
                    0.0,
                    &mut c[i * logical_batch_size],
                    b_shape[1].to_usize().unwrap() as isize,
                    1,
                );
            }
        }

        vec![Tensor { data: Box::new(c) }]
    }
}

/// Apply multiple unary ops in sequence, without having to reindex / rewrite to memory between each
#[derive(Debug, Default)]
pub struct UnaryFusionOptimizer;

impl GraphOptimizer for UnaryFusionOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        fn is_unary(op: &dyn Any) -> Option<fn(f32) -> f32> {
            if op.is::<Exp2>() {
                Some(|i| i.exp2())
            } else if op.is::<Log2>() {
                Some(|i| i.log2())
            } else if op.is::<Recip>() {
                Some(|i| i.recip())
            } else if op.is::<Sqrt>() {
                Some(|i| i.sqrt())
            } else if op.is::<Sin>() {
                Some(|i| i.sin())
            } else {
                None
            }
        }

        // Scan through unary sequential eliminations
        for id in graph.graph.node_indices().collect_vec() {
            if graph.no_delete.contains(&id) {
                continue;
            }
            let outgoing = graph
                .graph
                .edges_directed(id, petgraph::Direction::Outgoing)
                .map(|i| i.target())
                .collect_vec();
            if outgoing.len() != 1 {
                continue;
            }
            for outgoing_target in outgoing {
                let op = graph.graph.node_weight(id).unwrap();
                let other = graph.graph.node_weight(outgoing_target).unwrap();
                let mut replaced = false;
                if let Some(f) = is_unary(op.as_any()) {
                    if let Some(of) = is_unary(other.as_any()) {
                        // Unary -> Unary
                        *graph.graph.node_weight_mut(id).unwrap() =
                            Box::new(FusedUnary(vec![f, of]));
                        replaced = true;
                    } else if let Some(mut fused) =
                        other.as_any().downcast_ref::<FusedUnary>().cloned()
                    {
                        // Unary -> Fused
                        fused.0.insert(0, f);
                        *graph.graph.node_weight_mut(id).unwrap() = Box::new(fused);
                        replaced = true;
                    }
                } else if let Some(mut fused) = op.as_any().downcast_ref::<FusedUnary>().cloned() {
                    if let Some(of) = is_unary(other.as_any()) {
                        // Fused -> Unary
                        fused.0.push(of);
                        *graph.graph.node_weight_mut(id).unwrap() = Box::new(fused);
                        replaced = true;
                    } else if let Some(mut other_fused) =
                        other.as_any().downcast_ref::<FusedUnary>().cloned()
                    {
                        // Fused -> Fused
                        fused.0.append(&mut other_fused.0);
                        *graph.graph.node_weight_mut(id).unwrap() = Box::new(fused);
                        replaced = true;
                    }
                }
                if replaced {
                    // Remove other node
                    move_outgoing_edge(outgoing_target, id, &mut graph.graph);
                    move_references(
                        &mut graph.id_remap,
                        &mut graph.no_delete,
                        &mut graph.to_retrieve,
                        outgoing_target,
                        id,
                    );
                    graph.graph.remove_node(outgoing_target);
                }
            }
        }
    }
}

/// Multiple unary ops applied in sequence
#[derive(Debug, Clone, PartialEq)]
pub struct FusedUnary(Vec<fn(f32) -> f32>);

impl Operator for FusedUnary {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut t = match inp.pop().unwrap().0 {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t.clone(),
        };
        for a in t
            .data
            .as_any_mut()
            .downcast_mut::<Vec<f32>>()
            .unwrap()
            .iter_mut()
        {
            for f in &self.0 {
                *a = (f)(*a);
            }
        }

        vec![t]
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    #[test]
    fn test_cpu_matmul_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>("Input");
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let unoptimized_c = c.data();
        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
        cx.execute();

        assert_close_data(&c.data(), &unoptimized_c);
    }

    #[test]
    fn test_cpu_batch_matmul_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>("Input");
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let unoptimized_c = c.data();
        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
        cx.execute();

        assert_close_data(&c.data(), &unoptimized_c);
    }

    #[test]
    fn test_cpu_matmul_2d_2() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = cx.new_tensor::<R2<3, 4>>("Input");
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let unoptimized_c = c.data();
        cx.optimize(<(CPUOptimizer, GenericOptimizer)>::default());
        cx.execute();
        assert_close_data(&c.data(), &unoptimized_c);
    }
}
