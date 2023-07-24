use std::any::Any;

use itertools::Itertools;
use petgraph::visit::EdgeRef;

use crate::{op::Operator, prelude::*};

// Ops and optimizers specific to CPU execution

pub type CPUOptimizer = CPUMatMulOptimizer;

#[derive(Debug, Default)]
pub struct CPUMatMulOptimizer;

impl GraphOptimizer for CPUMatMulOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // Look for the matmul pattern
        for node in graph.graph.node_indices().collect_vec() {
            // Permute
            let Some((permute, permute_shape)) = graph.graph.node_weight(node) else {
                continue;
            };
            if permute.name() != "Permute" || permute_shape[0].len() != 2 {
                continue;
            }
            let permute_shape = permute_shape[0].clone();
            // Expand 1
            let mut dests = graph.get_dests(node);
            if dests.len() != 1 || dests[0].1 .0.name() != "Expand" || dests[0].1 .1[0].len() != 2 {
                continue;
            }
            let (expand_1, _) = dests.pop().unwrap();

            // Mul
            let mut dests = graph.get_dests(expand_1);
            if dests.len() != 1 || dests[0].1 .0.name() != "Mul" || dests[0].1 .1[0].len() != 3 {
                continue;
            }
            let (mul, _) = dests.pop().unwrap();

            // Expand 2
            let mut srcs = graph
                .get_sources(mul)
                .into_iter()
                .filter(|(i, _)| *i != expand_1)
                .collect_vec();
            if srcs.len() != 1 || srcs[0].1 .0.name() != "Expand" || srcs[0].1 .1[0].len() != 2 {
                continue;
            }
            let (expand_2, (_, s)) = srcs.pop().unwrap();
            let expand_2_shape = s[0].clone();

            let mut dests = graph.get_dests(mul);
            if dests.len() != 1
                || dests[0].1 .0.name() != "SumReduce"
                || dests[0].1 .1[0].len() != 3
            {
                continue;
            }
            let (sum_reduce, _) = dests.pop().unwrap();

            // We should check the dim of the sum reduce to make sure it's 2

            if graph.no_delete.contains(&node)
                || graph.no_delete.contains(&expand_1)
                || graph.no_delete.contains(&expand_2)
                || graph.no_delete.contains(&mul)
            {
                // One of these nodes is marked to not delete, we can't remove them
                continue;
            }

            let (input_0, _) = graph.get_sources(expand_2).pop().unwrap();
            let (input_1, _) = graph.get_sources(node).pop().unwrap();

            // Now we have a verified matmul, let's replace it with the CPUMatMul2D op
            let new_op = graph
                .add_op(CPUMatMul2D)
                .input(input_0, expand_2_shape)
                .input(input_1, permute_shape)
                .finish();

            // Create edges to dests
            for (weight, dest) in graph
                .graph
                .edges_directed(sum_reduce, petgraph::Direction::Outgoing)
                .map(|e| (*e.weight(), e.target()))
                .collect_vec()
            {
                graph.graph.add_edge(new_op, dest, weight);
            }
            Graph::move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                sum_reduce,
                new_op,
            );

            // Remove the old ops
            graph.graph.remove_node(expand_1);
            graph.graph.remove_node(expand_2);
            graph.graph.remove_node(node);
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}

#[derive(Debug)]
pub struct CPUMatMul2D;

impl Operator for CPUMatMul2D {
    fn name(&self) -> &'static str {
        "CPUMatMul2D"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        let a_shape = inp[0].shape.shape();
        let b_shape = inp[1].shape.shape();
        let a_strides = &inp[0].shape.views.last().unwrap().strides;
        let b_strides = &inp[1].shape.views.last().unwrap().strides;
        let a_data = inp[0].data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let b_data = inp[1].data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut c = vec![0.; a_shape[0] * b_shape[1]];
        unsafe {
            matrixmultiply::sgemm(
                a_shape[0],
                a_shape[1],
                b_shape[1],
                1.0,
                &a_data[0],
                a_strides[0] as isize,
                a_strides[1] as isize,
                &b_data[0],
                b_strides[0] as isize,
                b_strides[1] as isize,
                0.0,
                &mut c[0],
                b_shape[1] as isize,
                1,
            );
        }

        Tensor {
            data: Box::new(c),
            shape: ShapeTracker::new(vec![a_shape[0], b_shape[1]]),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    #[test]
    fn test_cpu_matmul_2_d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>();
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        cx.execute();

        let unoptimized_c = c.retrieve().unwrap();

        cx.optimize(<(CPUOptimizer, GeneralOpt)>::default());
        cx.execute();

        assert_close_data(
            &c.retrieve().unwrap().real_data().unwrap(),
            &unoptimized_c.real_data().unwrap(),
        );
    }
}
