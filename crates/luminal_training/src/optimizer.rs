use luminal::prelude::*;

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
    GraphTensor<()>,
) {
    let mut opt_graph = Graph::new();
    let lr = opt_graph.tensor().set(3e-4).keep();
    let mut old_weights = vec![];
    let mut gradients = vec![];
    let mut new_weights = vec![];
    for (_, param) in grads {
        let mut old_weight = opt_graph.named_tensor::<()>("old");
        old_weight.shape = *param;
        let mut gradient = opt_graph.named_tensor::<()>("grad");
        gradient.shape = *param;

        // SGD
        let new_weight = old_weight - (gradient * lr.expand_to(*param));
        new_weight.keep();

        old_weights.push(old_weight.id);
        gradients.push(gradient.id);
        new_weights.push(new_weight.id);
    }

    (old_weights, gradients, new_weights, opt_graph, lr)
}
