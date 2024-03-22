use std::collections::VecDeque;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    op::{
        Add, Constant, Exp2, Function, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Sin, Sqrt,
        SumReduce,
    },
    prelude::*,
};

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
        let mut grad_shape = graph
            .edges_directed(self.1, Direction::Incoming)
            .find_map(|e| e.weight().as_data().map(|(_, _, s)| s))
            .unwrap();
        // If it's a reduction, the loss output will have that dimension removed. Otherwise just use the contiguous version
        if let Some(SumReduce(dim)) = graph.try_get_op(self.1) {
            grad_shape.remove_dim(*dim);
        } else if let Some(MaxReduce(dim)) = graph.try_get_op(self.1) {
            grad_shape.remove_dim(*dim);
        }
        grad_shape = grad_shape.contiguous();
        chained_grads.insert(
            self.1,
            (
                graph
                    .add_op(Constant(
                        crate::op::ConstantValue::Float(1.0),
                        &graph.dyn_map,
                    ))
                    .finish(),
                grad_shape,
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
            // Get input tensors
            let inps = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, (a, _, _))| *a)
                .map(|(a, (_, _, b))| (a, b))
                .map(|(n, s)| GraphTensor::<()>::from_id(n, s, graph_ref))
                .collect::<Vec<_>>();
            let (prev_grad_node, mut prev_grad_shape) = chained_grads[&fwd_node];
            // If op is a reduction, we must add the dimension back
            if let Some(SumReduce(dim)) = op.downcast_ref() {
                prev_grad_shape.expand(*dim, inps[0].shape.shape()[*dim].clone().into());
            } else if let Some(MaxReduce(dim)) = op.downcast_ref() {
                prev_grad_shape.expand(*dim, inps[0].shape.shape()[*dim].clone().into());
            }
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
                    add_grad_to_map(inps[0].id, prev_grad_node, prev_grad_shape);
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
    use super::Module;

    crate::test_imports!();

    fn get_vec(id: NodeIndex, cx: &mut Graph) -> &Vec<f32> {
        cx.get_tensor_ref(id, 0)
            .unwrap()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap()
    }
    #[test]
    fn test_autograd_linear() {
        let mut cx = Graph::new();
        let model = crate::nn::Linear::<2, 1>::initialize(&mut cx);
        model.weight.set([[2.], [3.]]);
        let input = cx.named_tensor::<R1<2>>("Input").set([10., 5.]);
        let output = model.forward(input).retrieve();

        let grads = cx.compile(Autograd::new(state_set(&model), output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        assert_exact(get_vec(grads[0], &mut cx), &[10.0, 5.0]);
        assert_exact(&output.data(), &[35.0]);
    }
}
