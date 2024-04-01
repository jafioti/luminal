use luminal::prelude::*;

/// [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
///
/// This computes `(prediction - target).square().mean()`.
pub fn mse_loss<S: Shape>(prediction: GraphTensor<S>, target: GraphTensor<S>) -> GraphTensor<()> {
    (prediction - target).square().mean_reduce()
}

/// [Root Mean square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation).
///
/// This computes `(prediction - target).square().mean().sqrt()`
pub fn rmse_loss<S: Shape>(prediction: GraphTensor<S>, target: GraphTensor<S>) -> GraphTensor<()> {
    mse_loss(prediction, target).sqrt()
}

/// [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error).
///
/// This computes `(prediction - target).abs().mean()`
pub fn mae_loss<S: Shape>(prediction: GraphTensor<S>, target: GraphTensor<S>) -> GraphTensor<()> {
    (prediction - target).abs().mean_reduce()
}

/// [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < delta`: `0.5 * (x - y)^2`
/// 2. otherwise: `delta * (|x - y| - 0.5 * delta)`
pub fn huber_loss<S: Shape>(
    prediction: GraphTensor<S>,
    target: GraphTensor<S>,
    delta: impl Into<f32>,
) -> GraphTensor<()> {
    let delta: f32 = delta.into();
    let abs_error = (prediction - target).abs();
    let delta_tensor = prediction.graph().constant(delta);
    let huber_error = (0.5 * (prediction - target).square())
        * abs_error.less_than(delta_tensor.expand())
        + (delta * (abs_error - 0.5 * delta)) * abs_error.greater_than_equal(delta_tensor.expand());
    huber_error.mean_reduce()
}

/// Smooth l1 loss (closely related to [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss))
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < beta`: `0.5 * (x - y)^2 / beta`
/// 2. otherwise: `|x - y| - 0.5 * beta`
pub fn smooth_l1_loss<S: Shape>(
    prediction: GraphTensor<S>,
    target: GraphTensor<S>,
    delta: impl Copy + Into<f32>,
) -> GraphTensor<()> {
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
pub fn cross_entropy_with_logits_loss<S: Shape>(
    logits: GraphTensor<S>,
    target_probabilities: GraphTensor<S>,
) -> GraphTensor<()> {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(logits.shape.shape().last().unwrap());
    let probs = logits.log_softmax::<S::LastAxis>();
    (-(probs * target_probabilities).mean_reduce()) / inv_last_axis_numel
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
pub fn kl_div_with_logits_loss<S: Shape>(
    logits: GraphTensor<S>,
    target_probabilities: GraphTensor<S>,
) -> GraphTensor<()> {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(logits.shape.shape().last().unwrap());
    let probs = logits.log_softmax::<S::LastAxis>();
    (-((probs - target_probabilities.ln()) * target_probabilities).mean_reduce())
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
pub fn binary_cross_entropy_with_logits_loss<S: Shape>(
    logits: GraphTensor<S>,
    target_probabilities: GraphTensor<S>,
) -> GraphTensor<()> {
    let bce = (1.0 - target_probabilities) * logits + (1.0 + (-logits).exp()).ln();
    bce.mean_reduce()
}
