use cudarc::driver::*;
use cudarc::nvrtc::compile_ptx;

use std::thread;

const KERNEL_SRC: &str = "
extern \"C\" __global__ void hello_world(int i) {
    printf(\"Hello from the cuda kernel in thread %d\\n\", i);
}
";

fn main() -> Result<(), DriverError> {
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    {
        // Option 1: use the same device on each thread.
        // This requires calling the CudaDevice::bind_to_thread() method.
        // Note that all kernels are submitted to the same stream/context,
        // so the kernels will still execute in sequentially in the order
        // they are submitted to the gpu.
        let dev = CudaDevice::new(0)?;
        let ptx = compile_ptx(KERNEL_SRC).unwrap();
        dev.load_ptx(ptx, "kernel", &["hello_world"])?;

        // explicit borrow so we don't have to re-clone the device for each thread
        let dev = &dev;

        thread::scope(move |s| {
            for i in 0..10i32 {
                s.spawn(move || {
                    // NOTE: this is the important call to have
                    // without this, you'll get a CUDA_ERROR_INVALID_CONTEXT
                    dev.bind_to_thread()?;
                    let f = dev.get_func("kernel", "hello_world").unwrap();
                    unsafe { f.launch(cfg, (i,)) }
                });
            }
        });
    }

    {
        // Option 2: create a new device in each thread
        // This requires loading the PTX for each device, since they won't
        // share a loaded modules on the Rust side of things.
        let ptx = compile_ptx(KERNEL_SRC).unwrap();

        thread::scope(|s| {
            for i in 0..10i32 {
                let ptx = ptx.clone();
                s.spawn(move || {
                    let dev = CudaDevice::new(0)?;
                    dev.load_ptx(ptx, "kernel", &["hello_world"])?;
                    let f = dev.get_func("kernel", "hello_world").unwrap();
                    unsafe { f.launch(cfg, (i + 100,)) }
                });
            }
        });
    }

    Ok(())
}
