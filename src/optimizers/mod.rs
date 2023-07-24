/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
mod generic;
pub use generic::*;
mod cpu;
pub use cpu::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
