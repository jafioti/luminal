/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
mod general;
pub use general::*;
mod cpu;
pub use cpu::*;
mod cuda;
pub use cuda::*;
