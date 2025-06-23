use luminal::prelude::*;

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (Old weight inputs, Gradient inputs, New weight outputs, Optimizer Graph, Learning Rate Tensor)
pub fn sgd(
    grads: &[(NodeIndex, ShapeTracker)],
) -> (
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Graph,
    GraphTensor,
) {
    let mut opt_graph = Graph::new();
    let (old_weights, gradients): (Vec<NodeIndex>, Vec<NodeIndex>) = grads
        .iter()
        .map(|_| (opt_graph.tensor(1).id, opt_graph.tensor(1).id))
        .unzip();

    let (new_weights, lr) = sgd_on_graph(
        &mut opt_graph,
        &old_weights,
        &gradients
            .iter()
            .zip(grads)
            .map(|(a, (_, b))| (*a, *b))
            .collect::<Vec<_>>(),
    );
    (old_weights, gradients, new_weights, opt_graph, lr)
}

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (New weight outputs, Learning Rate Tensor)
pub fn sgd_on_graph(
    graph: &mut Graph,
    old_weights: impl ToIds,
    grads: &[(NodeIndex, ShapeTracker)],
) -> (Vec<NodeIndex>, GraphTensor) {
    let lr = graph.named_tensor("Learning Rate", 1).set(3e-4).keep(); // Karpathy constant
    let mut new_weights = vec![];
    for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weights.to_ids()) {
        let old_weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
        let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);

        // SGD
        let new_weight = old_weight - (gradient * lr.expand(grad_shape));
        new_weight.keep();

        new_weights.push(new_weight.id);
    }

    (new_weights, lr)
}

// /// Implements the [Adam](https://arxiv.org/abs/1412.6980) algorithm.
// pub fn adam(grads: &[(NodeIndex, ShapeTracker)]) {}
