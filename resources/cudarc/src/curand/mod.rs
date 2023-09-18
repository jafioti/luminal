//! Wrappers around the [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
//! in three levels. See crate documentation for description of each.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
