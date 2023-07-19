pub mod graph;
pub mod graph_tensor;
pub mod op;
pub mod optimizer;
pub mod shape;
pub mod tensor;

#[cfg(test)]
mod tests;

// struct Linear<const I: usize, const O: usize> {
//     weight: Tensor<(Const<I>, Const<O>)>,
// }

// impl<'a, const I: usize, const O: usize> Linear<I, O> {
//     fn new() -> Self {
//         Self {
//             weight: Tensor::new(vec![1.0; I * O]),
//         }
//     }

//     fn forward(&'a self, input: Graph<'a, Const<I>, Const<I>>) -> Graph<'a, Const<I>, Const<O>> {
//         input.matmul(&self.weight)
//     }
// }
