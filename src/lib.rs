mod core;
pub use crate::core::*;
pub mod nn;
pub mod optimizers;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::graph::*;
    pub use crate::graph_tensor::*;
    pub use crate::module::*;
    pub use crate::optimizer::*;
    pub use crate::optimizers::*;
    pub use crate::serialization::*;
    pub use crate::shape::*;
    pub use crate::tensor::*;
}
