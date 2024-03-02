//! Wrappers around the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html),
//! in three levels. See crate documentation for description of each.
//!
//! # safe api usage
//!
//! 1. Instantiate a [CudaDevice]:
//!
//! ```rust
//! # use cudarc::driver::*;
//! let device = CudaDevice::new(0).unwrap();
//! ```
//!
//! 2. Allocate device memory with host data with [CudaDevice::htod_copy()], [CudaDevice::alloc_zeros()],
//! or [CudaDevice::htod_sync_copy()].
//!
//! You can also copy data to CudaSlice using [CudaDevice::htod_sync_copy_into()]
//!
//! ```rust
//! # use cudarc::driver::*;
//! # let device = CudaDevice::new(0).unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros(10).unwrap();
//! let b_dev: CudaSlice<f32> = device.htod_copy(vec![0.0; 10]).unwrap();
//! let c_dev: CudaSlice<f32> = device.htod_sync_copy(&[1.0, 2.0, 3.0]).unwrap();
//! ```
//!
//! 3. Transfer to host memory with [CudaDevice::sync_reclaim()], [CudaDevice::dtoh_sync_copy()],
//! or [CudaDevice::dtoh_sync_copy_into()]
//!
//! ```rust
//! # use cudarc::driver::*;
//! # use std::rc::Rc;
//! # let device = CudaDevice::new(0).unwrap();
//! let a_dev: CudaSlice<f32> = device.alloc_zeros(10).unwrap();
//! let mut a_buf: [f32; 10] = [1.0; 10];
//! device.dtoh_sync_copy_into(&a_dev, &mut a_buf);
//! assert_eq!(a_buf, [0.0; 10]);
//! let a_host: Vec<f32> = device.sync_reclaim(a_dev).unwrap();
//! assert_eq!(&a_host, &[0.0; 10]);
//! ```
//!
//! ## Mutating device memory - [CudaFunction]
//!
//! See [LaunchAsync] and [CudaFunction].
//!
//! In order to mutate device data, you need to use cuda kernels.
//!
//! Loading kernels is done with [CudaDevice::load_ptx()]
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! let device = CudaDevice::new(0).unwrap();
//! device.load_ptx(ptx, "module_name", &["my_function"]).unwrap();
//! ```
//!
//! Retrieve the function using the registered module name & actual function name:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDevice::new(0).unwrap();
//! # device.load_ptx(ptx, "module_name", &["my_function"]).unwrap();
//! let func: CudaFunction = device.get_func("module_name", "my_function").unwrap();
//! ```
//!
//! Asynchronously execute the kernel:
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDevice::new(0).unwrap();
//! # device.load_ptx(ptx, "module_name", &["my_function"]).unwrap();
//! # let func: CudaFunction = device.get_func("module_name", "my_function").unwrap();
//! let mut a = device.alloc_zeros::<f32>(10).unwrap();
//! let cfg = LaunchConfig::for_num_elems(10);
//! unsafe { func.launch(cfg, (&mut a,)) }.unwrap();
//! ```
//!
//! Note: Launching kernels is **extremely unsafe**. See [LaunchAsync] for more info.
//!
//! ## Sub slices of [CudaSlice]
//!
//! For some operations, it is necessary to only operate on a small part of a single [CudaSlice].
//! For example, the slice may represent a batch of items, and you want to run separate kernels
//! on each of the items in the batch.
//!
//! Use [CudaSlice::try_slice()] and [CudaSlice::try_slice_mut()] for this. The returned
//! views ([CudaView] and [CudaViewMut] hold references to the owning [CudaSlice],
//! so rust's ownership system handles safety here.
//!
//! These view structs can be used with [CudaFunction].
//!
//! ```rust
//! # use cudarc::{driver::*, nvrtc::*};
//! # let ptx = compile_ptx("extern \"C\" __global__ void my_function(float *out) { }").unwrap();
//! # let device = CudaDevice::new(0).unwrap();
//! # device.load_ptx(ptx, "module_name", &["my_function"]).unwrap();
//! let mut a: CudaSlice<f32> = device.alloc_zeros::<f32>(3 * 10).unwrap();
//! for i_batch in 0..3 {
//!     let mut a_sub_view: CudaViewMut<f32> = a.try_slice_mut(i_batch * 10..).unwrap();
//!     let f: CudaFunction = device.get_func("module_name", "my_function").unwrap();
//!     let cfg = LaunchConfig::for_num_elems(10);
//!     unsafe { f.launch(cfg, (&mut a_sub_view,)) }.unwrap();
//! }
//! ```
//!
//! #### A note on implementation
//!
//! It would be possible to re-use [CudaSlice] itself for sub-slices, however that would involve adding
//! another structure underneath the hood that is wrapped in an [std::sync::Arc] to minimize data cloning.
//! Overall it seemed more complex than the current implementation.
//!
//! # Multi threading
//!
//! In order to use a [CudaDevice] on multiple threads, you must call [CudaDevice::bind_to_thread] on
//! each thread **before you use the device**.
//!
//! # Safety
//!
//! There are a number of aspects to this, but at a high level this API utilizes [std::sync::Arc] to
//! control when [CudaDevice] can be dropped.
//!
//! ### Context/Stream lifetimes
//!
//! The first part of safety is ensuring that [crate::driver::sys::CUcontext],
//! [crate::driver::sys::CUdevice], and [crate::driver::sys::CUstream] all
//! live the required amount of time (i.e. device outlives context, which outlives stream).
//!
//! This is accomplished by putting all of them inside one struct, the [CudaDevice]. There are other ways,
//! such as adding newtypes that carry lifetimes with them, but this approach was chosen to make working
//! with device pointers easier.
//!
//! Additionally, [CudaDevice] implements [Drop] as releasing all the data from the device in
//! the expected way.
//!
//! ### Device Data lifetimes
//!
//! The next part of safety is ensuring that [CudaSlice] do not outlive
//! the [CudaDevice]. For usability, each [CudaSlice] owns an `Arc<CudaDevice>`
//! to ensure the device stays alive.
//!
//! Additionally we don't want to double free any device pointers, so free is only
//! called when the device pointer is dropped. Thanks rust!
//!
//! ### Host and Device Data lifetimes
//!
//! Each device allocation can be associated with a host allocation. We want to ensure
//! that these have the same lifetimes *when copying data between them*.
//!
//! This is done via the various copy methods. Methods that don't take ownership
//! of the host data need to be executed synchronously, while the methods own the reference.
//! Methods that do own the host data can be executed synchronously.
//!
//! ### Single stream operations
//!
//! The next part of safety is ensuring that all operations happen on a single stream.
//! This ensures that data isn't mutated by more than 1 stream at a time, and also
//! ensures data isn't used before allocated, or used after free.
//!
//! Another important aspect of this is ensuring that mutability in an async setting
//! is sound, and something can't be freed while it's being used in a kernel.
//!
//! To this end every operation by default happens on the same stream.
//!
//! Multi stream is supported via [CudaStream], however it automatically
//! synchronizes with the main stream on creation & on drop. It is still possible
//! to be unsafe in a multi stream context though.

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
