//! Wrappers around the [cublas API](https://docs.nvidia.com/cuda/cublas/index.html),
//! in three levels. See crate documentation for description of each.

#[cfg(feature = "f16")]
pub mod half;
pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
