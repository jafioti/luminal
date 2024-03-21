use std::collections::VecDeque;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};
use tinyvec::ArrayVec;

use crate::{
    op::{
        Add, Constant, Exp2, Function, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Sin, Sqrt,
        SumReduce,
    },
    prelude::*,
};

use self::symbolic::Expression;

#[derive(Clone, Debug)]
pub struct Autograd(FxHashSet<NodeIndex>, NodeIndex);

impl Autograd {
    pub fn new<W: ToIds, L: ToId>(params: W, loss: L) -> Self {
        Self(params.to_ids().into_iter().collect(), loss.to_id())
    }
}

fn build_dfs_set(
    stack: &mut Vec<NodeIndex>,
    graph: &MainGraph,
    direction: Direction,
) -> FxHashSet<NodeIndex> {
    let mut set = FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !set.contains(&n) {
            set.insert(n);
            stack.extend(
                graph
                    .edges_directed(n, direction)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| match direction {
                        Direction::Incoming => e.source(),
                        Direction::Outgoing => e.target(),
                    }),
            );
        }
    }
    set
}

impl Compiler for Autograd {
    type Output = Vec<NodeIndex>;
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) -> Vec<NodeIndex> {
        // Build up valid set for nodes we want to pay attention to (everything outside of this set doesn't matter)
        let forward_set = build_dfs_set(
            &mut self.0.clone().into_iter().collect(),
            graph,
            Direction::Outgoing,
        );
        let backward_set = build_dfs_set(&mut vec![self.1], graph, Direction::Incoming);
        let valid_set = forward_set
            .intersection(&backward_set)
            .copied()
            .collect::<FxHashSet<_>>();

        // We have the last loss node, now let's backprop through everything to get the gradient graph
        // Referse bfs
        let mut bfs_queue = VecDeque::new();
        bfs_queue.push_back(self.1);
        let mut chained_grads = FxHashMap::default();
        // Add loss gradient
        let loss_shape = graph
            .edges_directed(self.1, Direction::Incoming)
            .next()
            .unwrap()
            .weight()
            .as_data()
            .unwrap()
            .2;
        chained_grads.insert(
            self.1,
            (
                graph
                    .add_op(Constant(
                        crate::op::ConstantValue::Float(1.0),
                        &graph.dyn_map,
                    ))
                    .finish(),
                loss_shape,
            ),
        );
        while let Some(fwd_node) = bfs_queue.pop_front() {
            if !valid_set.contains(&fwd_node) {
                continue;
            }
            // Check if the node is undifferentiable
            let graph_ref: *mut Graph = graph;
            let op = graph.node_weight(fwd_node).unwrap().as_any();
            if op.is::<Function>() {
                continue;
            }
            if op.is::<Mod>() || op.is::<LessThan>() {
                if self.0.contains(&fwd_node) {
                    panic!(
                        "Node {} is marked as a weight but is undifferentiable: {:?}",
                        fwd_node.index(),
                        graph.node_weight(fwd_node).unwrap()
                    );
                }
                continue;
            }
            // Determine reverse shapes
            let reverse_shapes = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|(a, _, _)| *a)
                .map(|(_, _, mut sh)| {
                    // Reset permutes
                    let mut dims = vec![0; sh.len()];
                    for i in 0..sh.len() {
                        dims[sh.indexes[i]] = i;
                    }
                    for (d, i) in dims.iter().zip(sh.indexes.iter_mut()) {
                        *i = *d;
                    }
                    // Reset slices
                    let mut new_padding = ArrayVec::<[(Expression, Expression); 6]>::new();
                    let zero = Expression::from(0);
                    for (i, (a, b)) in sh.slices.iter().enumerate() {
                        let l = if *a != zero { *a } else { zero };
                        let r = if *b != Expression::from(i32::MAX) && *b != sh.dims[i] {
                            sh.dims[i] - b
                        } else {
                            zero
                        };
                        new_padding.push((l, r));
                    }
                    // Reset padding
                    let mut new_slices = ArrayVec::<[(Expression, Expression); 6]>::new();
                    for (i, (a, b)) in sh.padding.iter().enumerate() {
                        let l = if *a != zero { *a } else { zero };
                        let r = if *b != zero {
                            *b - sh.dims[i]
                        } else {
                            Expression::from(i32::MAX)
                        };
                        new_slices.push((l, r));
                    }
                    sh.padding = new_padding;
                    sh.slices = new_slices;
                    // Reset expands
                    for i in (0..sh.len()).rev() {
                        if sh.fake[i] {
                            sh.remove_dim(i);
                        }
                    }
                    sh
                })
                .collect::<Vec<_>>();
            let inps = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, (a, _, _))| *a)
                .map(|(a, _)| a)
                .zip(reverse_shapes)
                .map(|(n, s)| GraphTensor::<()>::from_id(n, s, graph_ref))
                .collect::<Vec<_>>();
            let (prev_grad_node, prev_grad_shape) = chained_grads[&fwd_node];
            let prev_grad = GraphTensor::<()>::from_id(prev_grad_node, prev_grad_shape, graph_ref);
            let mut add_grad_to_map = |fwd_id, grad_id, grad_shape| {
                if let Some((existing_grad_node, existing_grad_shape)) =
                    chained_grads.get(&fwd_id).copied()
                {
                    let grad = GraphTensor::<()>::from_id(
                        existing_grad_node,
                        existing_grad_shape,
                        graph_ref,
                    );
                    let new_grad = prev_grad + grad;
                    chained_grads.insert(fwd_id, (new_grad.id, grad_shape));
                } else {
                    chained_grads.insert(fwd_id, (grad_id, grad_shape));
                }
            };
            if op.is::<Add>() {
                // f(a, b) = a + b
                // df/da = 1
                // df/db = 1
                if valid_set.contains(&inps[0].id) {
                    add_grad_to_map(inps[0].id, prev_grad_node, inps[0].shape);
                }
                if valid_set.contains(&inps[1].id) {
                    add_grad_to_map(inps[1].id, prev_grad_node, inps[1].shape);
                }
            } else if op.is::<Mul>() {
                // f(a, b) = a * b
                // df/da = b
                // df/db = a
                if valid_set.contains(&inps[0].id) {
                    let a_grad = inps[1] * prev_grad;
                    add_grad_to_map(inps[0].id, a_grad.id, inps[0].shape);
                }
                if valid_set.contains(&inps[1].id) {
                    let b_grad = inps[0] * prev_grad;
                    add_grad_to_map(inps[1].id, b_grad.id, inps[1].shape);
                }
            } else if op.is::<SumReduce>() {
                // f(x) = sum_reduce(x)
                // f'(x) = 1
                if valid_set.contains(&inps[0].id) {
                    add_grad_to_map(inps[0].id, prev_grad_node, inps[0].shape);
                }
            } else if let Some(op) = op.downcast_ref::<MaxReduce>().cloned() {
                // f(x) = sum_reduce(x)
                // f'(x) = x == sum_reduce(x)
                if valid_set.contains(&inps[0].id) {
                    let reduced = graph
                        .add_op(MaxReduce(op.0))
                        .input(inps[0].id, 0, inps[0].shape)
                        .finish();
                    let mut shape = inps[0].shape;
                    let size = shape.remove_dim(op.0);
                    shape.expand(op.0, size);
                    let reduced = GraphTensor::<()>::from_id(reduced, shape, graph_ref);
                    let a_grad = inps[0].equals(reduced);
                    add_grad_to_map(a_grad.id, prev_grad_node, inps[0].shape);
                }
            } else {
                if !valid_set.contains(&inps[0].id) {
                    continue;
                }
                let local_grad = if op.is::<Log2>() {
                    // f(x) = log2(x)
                    // f'(x) = 1 / (x * ln(2))
                    1.0 / (inps[0] * 2_f32.ln())
                } else if op.is::<Exp2>() {
                    // f(x) = exp2(x)
                    // f'(x) = exp2(x) * ln(2)
                    inps[0] * 2_f32.ln()
                } else if op.is::<Sin>() {
                    // f(x) = sin(x)
                    // f'(x) = cos(x)
                    inps[0].cos()
                } else if op.is::<Sqrt>() {
                    // f(x) = sqrt(x)
                    // f'(x) = 1 / (2 * sqrt(x))
                    1.0 / (2.0 * inps[0].sqrt())
                } else if op.is::<Recip>() {
                    // f(x) = 1 / x
                    // f'(x) = -1 / x**2
                    -1.0 / inps[0].pow(2.0)
                } else {
                    unreachable!()
                };
                let new_grad = local_grad * prev_grad; // Chain rule
                add_grad_to_map(inps[0].id, new_grad.id, new_grad.shape);
            }

            // Continue bfs
            for node in inps {
                bfs_queue.push_back(node.id);
            }
        }

        // Create a gradient array to match 1-1 with the weight array passed in
        let mut grad_array = vec![];
        for weight in &self.0 {
            let grad = chained_grads[weight].0;
            graph.no_delete.insert(grad);
            grad_array.push(grad);
        }
        grad_array
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    fn get_scalar_data(id: NodeIndex, cx: &mut Graph) -> f32 {
        cx.get_tensor(id, 0)
            .unwrap()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap()[0]
    }
    #[test]
    fn test_autograd() {
        let mut cx = Graph::new();
        let weight = cx.named_tensor::<R0>("Weight").set(2.);
        let bias = cx.named_tensor::<R0>("Bias").set(-3.);
        let input = cx.named_tensor::<R0>("Input").set(10.);
        let output = input * weight + bias;

        let grads = cx.compile(Autograd::new((weight, bias), output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        assert_exact(&[get_scalar_data(grads[0], &mut cx)], &[10.0]);
        assert_exact(&[get_scalar_data(grads[1], &mut cx)], &[1.0]);
    }
}
