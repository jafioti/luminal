mod core;
pub use crate::core::*;
pub mod nn;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::graph::*;
    pub use crate::graph_tensor::*;
    pub use crate::module::*;
    pub use crate::optimizer::*;
    pub use crate::shape::*;
    pub use crate::tensor::*;
}

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
