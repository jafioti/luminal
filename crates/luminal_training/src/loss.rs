use luminal::prelude::*;

/// [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
///
/// This computes `(prediction - target).square().mean()`.
pub fn mse_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    (prediction - target)
        .square()
        .mean(prediction.shape.all_axes())
}

/// [Root Mean square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation).
///
/// This computes `(prediction - target).square().mean().sqrt()`
pub fn rmse_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    mse_loss(prediction, target).sqrt()
}

/// [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error).
///
/// This computes `(prediction - target).abs().mean()`
pub fn mae_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    (prediction - target)
        .abs()
        .mean(prediction.shape.all_axes())
}

/// [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < delta`: `0.5 * (x - y)^2`
/// 2. otherwise: `delta * (|x - y| - 0.5 * delta)`
pub fn huber_loss(
    prediction: GraphTensor,
    target: GraphTensor,
    delta: impl Into<f32>,
) -> GraphTensor {
    let delta: f32 = delta.into();
    let abs_error = (prediction - target).abs();
    let delta_tensor = prediction.graph().constant(delta);
    let huber_error = (0.5 * (prediction - target).square())
        * abs_error.lt(delta_tensor.expand(abs_error.shape))
        + (delta * (abs_error - 0.5 * delta)) * abs_error.ge(delta_tensor.expand(abs_error.shape));
    huber_error.mean(huber_error.shape.all_axes())
}

/// Smooth l1 loss (closely related to [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss))
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < beta`: `0.5 * (x - y)^2 / beta`
/// 2. otherwise: `|x - y| - 0.5 * beta`
pub fn smooth_l1_loss(
    prediction: GraphTensor,
    target: GraphTensor,
    delta: impl Into<f32> + Copy,
) -> GraphTensor {
    huber_loss(prediction, target, delta) / delta.into()
}

/// [Cross entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression).
/// This computes: `-(logits.log_softmax() * target_probs).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not the
/// output from** [softmax()] or [log_softmax()] already.
///
/// ### Inputs
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probabilities`: Target containing probability vectors **NOT** class indices.
pub fn cross_entropy_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(*logits.shape.dims().last().unwrap());
    let probs = logits.log_softmax(logits.shape.last_axis());
    (-(probs * target_probabilities).mean(target_probabilities.shape.all_axes()))
        / inv_last_axis_numel
}

/// [KL Divergence loss](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
/// This computes `(target_probs * (target_probs.log() - logits.log_softmax())).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not the
/// output from** [softmax()] or [log_softmax()] already.
///
/// ### Inputs
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
pub fn kl_div_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(*logits.shape.dims().last().unwrap());
    let probs = logits.log_softmax(logits.shape.last_axis());
    (-((probs - target_probabilities.log()) * target_probabilities)
        .mean(target_probabilities.shape.all_axes()))
        / inv_last_axis_numel
}

/// [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)
/// With Logits in numerically stable way.
///
/// Computes `target_probs * log(sigmoid(logits)) + (1 - target_probs) * log(1 - sigmoid(logits))`
/// as `(1 - target_probs) * logits + log(1 + exp(-logits))`.
///
/// ### Inputs
/// - `logits` - unnormalized inputs. **NOT** output of sigmoid
/// - `target_probs` - target values between 0 and 1.
pub fn binary_cross_entropy_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let bce = (1.0 - target_probabilities) * logits + (1.0 + (-logits).exp()).log();
    bce.mean(bce.shape.all_axes())
}
