#![allow(non_snake_case)]
//! This file outlines a typical build process which can be used for more complex CUDA projects utilising this crate.
//! It does the following:
//!     1. Use a `build.rs` file to compile your CUDA code/project into a PTX file. Your CUDA code/project can be as complicated as you need them to be, including multiple files, with headers for your struct definitions, each kernel in it's own file, etc.
//!     2. The build process compiles the kernels into a PTX file, which is written to the output directory
//!     3. The build process then uses the `bindgen` crate to generate Rust bindings for the structs defined in your CUDA code
//!     4. In the `main.rs` code, the PTX code is included as a string via the `!include_str` macro, which is then compiled using the functions in this crate (detailed in previous examples)
//!
//! The advantages of having this build process for more complex CUDA projects:
//!     - You only need to define your structs once, in your CUDA code, and the Rust bindings are generated automatically
//!     - You have full intellisense for your CUDA code since they can be stored under a separate folder or even as part of a separate project
//!
//! There are two files in this example: `main.rs` and `build.rs`. You can reference them and add to your project accordingly. The `cuda` folder in this example gives a simple example of defining structs in a separate header, including creating a `wrapper.h` header for `bindgen`

use std::time::Instant;
use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl DeviceRepr for MyStruct {}
impl Default for MyStruct {
    fn default() -> Self{
        Self{ data: [0.0; 4]}
    }
}

// include the compiled PTX code as string
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/my_struct_kernel.ptx"));

fn main() -> Result<(), DriverError> {
    // setup GPU device
    let now = Instant::now();

    let gpu = CudaDevice::new(0)?;

    println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

    // compile ptx
    let now = Instant::now();

    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "my_module", &["my_struct_kernel"])?;

    println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

    // create data
    let now = Instant::now();

    let n = 10_usize;
    let my_structs = vec![MyStruct { data: [1.0; 4] }; n];

    // copy to GPU
    let gpu_my_structs = gpu.htod_copy(my_structs)?;

    println!("Time taken to initialise data: {:.2?}", now.elapsed());

    let now = Instant::now();

    let f = gpu.get_func("my_module", "my_struct_kernel").unwrap();

    unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&gpu_my_structs, n)) }?;

    println!("Time taken to call kernel: {:.2?}", now.elapsed());

    let my_structs = gpu.sync_reclaim(gpu_my_structs)?;

    assert!(my_structs.iter().all(|i| i.data == [1.0; 4]));

    Ok(())
}
