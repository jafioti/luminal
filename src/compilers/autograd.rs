use std::collections::VecDeque;

use itertools::Itertools;
use petgraph::{visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    op::{
        Add, Constant, ConstantValue, Exp2, Function, LessThan, Log2, MaxReduce, Mod, Mul, Recip,
        Sin, Sqrt, SumReduce,
    },
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct Autograd(Vec<NodeIndex>, NodeIndex);

impl Autograd {
    pub fn new<W: ToIds, L: ToId>(params: W, loss: L) -> Self {
        Self(params.to_ids(), loss.to_id())
    }
}

// Run dfs with a starting stack and record all encountered nodes in a set
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
        let Autograd(params, loss) = self;
        // Build up valid set for nodes we want to pay attention to (everything outside of this set doesn't matter)
        let forward_set = build_dfs_set(&mut params.clone(), graph, Direction::Outgoing);
        let backward_set = build_dfs_set(&mut vec![*loss], graph, Direction::Incoming);
        let valid_set: FxHashSet<_> = forward_set.intersection(&backward_set).copied().collect();

        // We have the last loss node, now let's backprop through everything to get the gradient graph
        // Referse bfs
        let mut bfs_queue = VecDeque::new();
        bfs_queue.push_back(*loss);
        let mut chained_grads = FxHashMap::default();
        // Add loss gradient
        let mut grad_shape = graph
            .edges_directed(*loss, Direction::Incoming)
            .find_map(|e| e.weight().as_data().map(|(_, _, s)| s))
            .unwrap()
            .contiguous();
        // If it's a reduction, the loss output will have that dimension removed. Otherwise just use the contiguous version
        if let Some(SumReduce(dim)) = graph.try_get_op(*loss) {
            grad_shape.remove_dim(*dim);
        } else if let Some(MaxReduce(dim)) = graph.try_get_op(*loss) {
            grad_shape.remove_dim(*dim);
        }
        chained_grads.insert(
            *loss,
            (
                graph
                    .add_op(Constant(ConstantValue::Float(1.0), &graph.dyn_map))
                    .finish(),
                grad_shape,
            ),
        );
        let weight_set = params.iter().copied().collect::<FxHashSet<_>>();
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
                assert!(
                    !weight_set.contains(&fwd_node),
                    "{fwd_node:?} is marked as a weight but is undifferentiable: {:?}",
                    graph.node_weight(fwd_node).unwrap()
                );
                continue;
            }

            // Differentiate through fwd_node to get gradients for it's sources
            // Get input tensors
            let inps = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, (a, _, _))| *a)
                .map(|(node, (_, _, sh))| GraphTensor::<()>::from_id(node, sh, graph_ref))
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
                if valid_set.contains(&inps[0].id) {
                    add_grad_to_map(inps[0].id, prev_grad_node, inps[0].shape);
                }
                // df/db = 1
                if valid_set.contains(&inps[1].id) {
                    add_grad_to_map(inps[1].id, prev_grad_node, inps[1].shape);
                }
            } else if op.is::<Mul>() {
                // f(a, b) = a * b
                // df/da = b
                if valid_set.contains(&inps[0].id) {
                    let a_grad = inps[1] * prev_grad;
                    add_grad_to_map(inps[0].id, a_grad.id, inps[0].shape);
                }
                // df/db = a
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
                // f(x) = max_reduce(x)
                // f'(x) = x == max_reduce(x)
                if valid_set.contains(&inps[0].id) {
                    // fwd_nod is already max_reduce(x)
                    let mut shape = inps[0].shape;
                    let size = shape.remove_dim(op.0);
                    shape.expand(op.0, size);
                    let reduced = GraphTensor::<()>::from_id(fwd_node, shape, graph_ref);
                    let new_grad = inps[0].equals(reduced) * prev_grad;
                    add_grad_to_map(inps[0].id, new_grad.id, new_grad.shape);
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
