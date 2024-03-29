use std::{any::TypeId, collections::VecDeque};

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
    pub fn new<W: ToIds>(params: W, loss: GraphTensor<()>) -> Self {
        Self(params.to_ids(), loss.id)
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
        let mut grads = FxHashMap::default();
        // Add loss gradient
        grads.insert(
            *loss,
            (
                graph
                    .add_op(Constant(ConstantValue::Float(1.0), &graph.dyn_map))
                    .finish(),
                ShapeTracker::new(&[]), // Assume scalar loss for now
            ),
        );
        let weight_set = params.iter().copied().collect::<FxHashSet<_>>();
        while let Some(fwd_node) = bfs_queue.pop_front() {
            if !valid_set.contains(&fwd_node) {
                continue;
            }
            // Check if the node is undifferentiable
            let graph_ref: *mut Graph = graph;
            let op = graph.node_weight(fwd_node).unwrap().as_any().type_id();
            if op == TypeId::of::<Function>() {
                continue;
            }
            if op == TypeId::of::<Mod>() || op == TypeId::of::<LessThan>() {
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
            let mut prev_grad = {
                let (id, sh) = grads[&fwd_node];
                GraphTensor::from_id(id, sh, graph_ref)
            };
            if op == TypeId::of::<Add>() {
                // f(a, b) = a + b
                // df/da = 1
                if valid_set.contains(&inps[0].id) {
                    add_grad(graph, prev_grad, inps[0], &mut grads);
                }
                // df/db = 1
                if valid_set.contains(&inps[1].id) {
                    add_grad(graph, prev_grad, inps[1], &mut grads);
                }
            } else if op == TypeId::of::<Mul>() {
                // f(a, b) = a * b
                // df/da = b
                if valid_set.contains(&inps[0].id) {
                    let grad = inps[1] * prev_grad;
                    add_grad(graph, grad, inps[0], &mut grads);
                }
                // df/db = a
                if valid_set.contains(&inps[1].id) {
                    let b_grad = inps[0] * prev_grad;
                    add_grad(graph, b_grad, inps[1], &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<SumReduce>(fwd_node)
                .cloned()
            {
                // f(x) = sum_reduce(x)
                // f'(x) = 1
                if valid_set.contains(&inps[0].id) {
                    prev_grad
                        .shape
                        .expand(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    add_grad(graph, prev_grad, inps[0], &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<MaxReduce>(fwd_node)
                .cloned()
            {
                // f(x) = max_reduce(x)
                // f'(x) = x == max_reduce(x)
                if valid_set.contains(&inps[0].id) {
                    // fwd_nod is already max_reduce(x)
                    prev_grad
                        .shape
                        .expand(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    let reduced = GraphTensor::<()>::from_id(fwd_node, prev_grad.shape, graph_ref);
                    let new_grad = inps[0].equals(reduced) * prev_grad;
                    add_grad(graph, new_grad, inps[0], &mut grads);
                }
            } else {
                if !valid_set.contains(&inps[0].id) {
                    continue;
                }
                let local_grad = if op == TypeId::of::<Log2>() {
                    // f(x) = log2(x)
                    // f'(x) = 1 / (x * ln(2))
                    1.0 / (inps[0] * 2_f32.ln())
                } else if op == TypeId::of::<Exp2>() {
                    // f(x) = exp2(x)
                    // f'(x) = exp2(x) * ln(2)
                    inps[0] * 2_f32.ln()
                } else if op == TypeId::of::<Sin>() {
                    // f(x) = sin(x)
                    // f'(x) = cos(x)
                    inps[0].cos()
                } else if op == TypeId::of::<Sqrt>() {
                    // f(x) = sqrt(x)
                    // f'(x) = 1 / (2 * sqrt(x))
                    1.0 / (2.0 * inps[0].sqrt())
                } else if op == TypeId::of::<Recip>() {
                    // f(x) = 1 / x
                    // f'(x) = -1 / x**2
                    -1.0 / inps[0].pow(2.0)
                } else {
                    unreachable!()
                };
                let new_grad = local_grad * prev_grad; // Chain rule
                add_grad(graph, new_grad, inps[0], &mut grads);
            }

            // Continue bfs
            for node in inps {
                bfs_queue.push_back(node.id);
            }
        }

        // Create a gradient array to match 1-1 with the weight array passed in
        let mut grad_array = vec![];
        for weight in &self.0 {
            let grad = grads[weight].0;
            graph.no_delete.insert(grad);
            grad_array.push(grad);
        }
        grad_array
    }
}

fn add_grad(
    graph: &mut Graph,
    mut grad: GraphTensor<()>,
    fwd: GraphTensor<()>,
    grad_map: &mut FxHashMap<NodeIndex, (NodeIndex, ShapeTracker)>,
) {
    // Reshape gradient to match the shape of the input source (before the input was reshaped)
    // Undo permutes
    for i in 0..fwd.shape.len() {
        grad.shape.indexes[fwd.shape.indexes[i]] = i;
    }

    // Undo expands (sum reduce)
    for i in (0..fwd.shape.len()).rev() {
        if fwd.shape.fake[fwd.shape.indexes[i]] {
            grad.id = graph
                .add_op(SumReduce(fwd.shape.indexes[i]))
                .input(grad.id, 0, grad.shape)
                .finish();
            grad.shape.remove_dim(fwd.shape.indexes[i]);
            grad.shape = grad.shape.contiguous();
        }
    }

    if let Some((existing_grad_node, existing_grad_shape)) = grad_map.get(&fwd.id).copied() {
        let grad = GraphTensor::<()>::from_id(grad.id, grad.shape, graph);
        let existing_grad =
            GraphTensor::<()>::from_id(existing_grad_node, existing_grad_shape, graph);
        let new_grad = grad + existing_grad;
        grad_map.insert(fwd.id, (new_grad.id, grad.shape));
    } else {
        grad_map.insert(fwd.id, (grad.id, grad.shape));
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn, prelude::Module};

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
    fn test_autograd_matmul() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("A").set([[2., 4.], [3., 1.]]);
        let input = cx.named_tensor("Input").set([10., 5.]);
        let output = (input.matmul(a)).sum_reduce();

        let grads = cx.compile(Autograd::new(a, output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let w1 = dev.tensor([[2., 4.], [3., 1.]]);
        let inp = dev.tensor([10., 5.]);
        let out = inp.trace(Gradients::leaky()).matmul(w1.clone()).sum();
        let d_grads = out.backward();

        assert_exact(get_vec(grads[0], &mut cx), &d_grads.get(&w1).as_vec());
    }

    #[test]
    fn test_autograd_mlp() {
        let mut cx = Graph::new();
        let model = <(nn::Linear<2, 2>, nn::ReLU, nn::Linear<2, 1>)>::initialize(&mut cx);
        model.0.weight.set([[2., 4.], [3., 1.]]);
        model.2.weight.set([[6.], [5.]]);
        let input = cx.named_tensor("Input").set([10., 5.]);
        let output = model.forward(input).sum_reduce();

        let grads = cx.compile(Autograd::new(params(model), output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let w1 = dev.tensor([[2., 4.], [3., 1.]]);
        let w2 = dev.tensor([[6.], [5.]]);
        let inp = dev.tensor([10., 5.]);
        let out = inp
            .trace(Gradients::leaky())
            .matmul(w1.clone())
            .relu()
            .matmul(w2.clone())
            .sum();
        let d_grads = out.backward();

        assert_exact(get_vec(grads[0], &mut cx), &d_grads.get(&w1).as_vec());
        assert_exact(get_vec(grads[1], &mut cx), &d_grads.get(&w2).as_vec());
    }
}
