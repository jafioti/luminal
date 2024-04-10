//! Wrappers around the [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
//! in three levels. See crate documentation for description of each.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
