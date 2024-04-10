//! Safe abstractions over:
//! 1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
//! 2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
//! 3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
//! 4. [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
//!
//! # crate organization
//!
//! Each of the modules for the above is organized into three levels:
//! 1. A `safe` module which provides safe abstractions over the `result` module
//! 2. A `result` which is a thin wrapper around the `sys` module to ensure all functions return [Result]
//! 3. A `sys` module which contains the raw FFI bindings
//!
//! | API | Safe | Result | Sys |
//! | --- | --- | --- | --- |
//! | driver | [driver::safe] | [driver::result] | [driver::sys] |
//! | cublas | [cublas::safe] | [cublas::result] | [cublas::sys] |
//! | cublaslt | [cublaslt::safe] | [cublaslt::result] | [cublaslt::sys] |
//! | nvrtc | [nvrtc::safe] | [nvrtc::result] | [nvrtc::sys] |
//! | curand | [curand::safe] | [curand::result] | [curand::sys] |
//! | cudnn | - | [cudnn::result] | [cudnn::sys] |
//!
//! # Core Concepts
//!
//! At the core is the [driver] API, which exposes a bunch of structs, but the main ones are:
//!
//! 1. [`driver::CudaDevice`] is a handle to a specific device ordinal (e.g. 0, 1, 2, ...)
//! 2. [`driver::CudaSlice<T>`], which represents a [`Vec<T>`] on the device, can be allocated
//!    using the aforementioned CudaDevice.
//!
//! Here is a table of similar concepts between CPU and Cuda:
//!
//! | Concept | CPU | Cuda |
//! | --- | --- | --- |
//! | Memory allocator | [`std::alloc::GlobalAlloc`] | [`driver::CudaDevice`] |
//! | List of values on heap | [`Vec<T>`] | [`driver::CudaSlice<T>`] |
//! | Slice | `&[T]` | [`driver::CudaView<T>`] |
//! | Mutable Slice  | `&mut [T]` | [`driver::CudaViewMut<T>`] |
//! | Function | [`Fn`] | [`driver::CudaFunction`] |
//! | Calling a function | `my_function(a, b, c)` | [`driver::LaunchAsync::launch()`] |
//! | Thread | [`std::thread::Thread`] | [`driver::CudaStream`] |
//!
//! # Combining the different APIs
//!
//! All the highest level apis have been designed to work together.
//!
//! ## nvrtc
//!
//! [`nvrtc::compile_ptx()`] outputs a [`nvrtc::Ptx`], which can
//! be loaded into a device with [`driver::CudaDevice::load_ptx()`].
//!
//! ## cublas
//!
//! [cublas::CudaBlas] can perform gemm operations using [`cublas::Gemm<T>`],
//! and [`cublas::Gemv<T>`]. Both of these traits can generically accept memory
//! allocated by the driver in the form of: [`driver::CudaSlice<T>`],
//! [`driver::CudaView<T>`], and [`driver::CudaViewMut<T>`].
//!
//! ## curand
//!
//! [curand::CudaRng] can fill a [`driver::CudaSlice<T>`] with random data, based on
//! one of its available distributions.
//!
//! # Combining safe/result/sys
//!
//! The result and sys levels are very inter-changeable for each API. However,
//! the safe apis don't necessarily allow you to mix in the result level. This
//! is to encourage going through the safe API when possible.
//!
//! **If you need some functionality that isn't present in the safe api, please
//! open a ticket.**

#![cfg_attr(feature = "no-std", no_std)]

#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;

#[cfg(feature = "cublas")]
pub mod cublas;
#[cfg(feature = "cublaslt")]
pub mod cublaslt;
#[cfg(feature = "cudnn")]
pub mod cudnn;
#[cfg(feature = "curand")]
pub mod curand;
#[cfg(feature = "driver")]
pub mod driver;
#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nvrtc")]
pub mod nvrtc;

pub mod types;
