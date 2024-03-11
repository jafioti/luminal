mod binary;
mod matmul;
mod other;

use std::any::Any;

use itertools::Itertools;
use petgraph::visit::EdgeRef;

use crate::{
    op::{Constant, ConstantValue, Exp2, InputTensor, Log2, Operator, Recip, Sin},
    prelude::*,
};

// Ops and compilers specific to CPU execution

pub type CPUCompiler = (
    matmul::MatMulCompiler,
    binary::SubtractionCompiler,
    binary::EqualCompiler,
    other::ARangeCompiler,
    binary::GatherCompiler,
    UnaryFusionCompiler,
);

pub(crate) fn constant(num: f32) -> SelectGraph {
    let mut n = op::<Constant>();
    n.check(move |o, _| {
        if let Some(c) = o.as_any().downcast_ref::<Constant>() {
            match c.0 {
                ConstantValue::Float(f) => f == num,
                _ => false,
            }
        } else {
            false
        }
    });
    n
}

/// Apply multiple unary ops in sequence, without having to reindex / rewrite to memory between each
#[derive(Debug, Default)]
pub struct UnaryFusionCompiler;

impl Compiler for UnaryFusionCompiler {
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        fn is_unary(op: &dyn Any) -> Option<fn(f32) -> f32> {
            if op.is::<Exp2>() {
                Some(|i| i.exp2())
            } else if op.is::<Log2>() {
                Some(|i| i.log2())
            } else if op.is::<Recip>() {
                Some(|i| i.recip())
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
                    move_outgoing_edge(outgoing_target, id, graph);
                    remap(outgoing_target, id, &mut ids, graph);
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
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut t = inp.pop().unwrap().0.cloned();
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
    use rand::{rngs::StdRng, SeedableRng};

    use crate::prelude::*;
    crate::test_imports!();

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let a = cx.tensor::<(Dyn<'M'>, Dyn<'K'>)>();
        let b = cx.tensor::<(Dyn<'K'>, Dyn<'N'>)>();
        let mut c = a.matmul(b).retrieve();

        cx.compile(CPUCompiler::default(), &mut c);

        let d_dev = Cpu::default();
        for m in (1..23).step_by(4) {
            for k in (1..35).step_by(3) {
                for n in (1..70).step_by(7) {
                    let mut rng = StdRng::seed_from_u64(0);
                    let a_data = random_vec_rng(m * k, &mut rng);
                    let b_data = random_vec_rng(k * n, &mut rng);
                    a.set_dyn(a_data.clone(), &[m, k]);
                    b.set_dyn(b_data.clone(), &[k, n]);

                    cx.execute();

                    let d_a = d_dev.tensor_from_vec(a_data, (m, k));
                    let d_b = d_dev.tensor_from_vec(b_data, (k, n));
                    let d_c = d_a.matmul(d_b);

                    assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
                    c.drop();
                }
            }
        }
    }

    #[test]
    fn test_cpu_matmul_2d_2() {
        let mut cx = Graph::new();
        let a = cx.tensor::<R2<2, 3>>();
        a.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let b = cx.tensor::<R2<3, 4>>();
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let mut c = a.matmul(b).retrieve();

        cx.execute();

        let unoptimized_c = c.data();
        cx.compile(CPUCompiler::default(), &mut c);
        cx.execute();
        assert_close(&c.data(), &unoptimized_c);
    }
}
