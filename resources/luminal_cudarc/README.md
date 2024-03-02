# cudarc: minimal and safe api over the cuda toolkit

[![](https://dcbadge.vercel.app/api/server/AtUhGqBDP5)](https://discord.gg/AtUhGqBDP5)
[![crates.io](https://img.shields.io/crates/v/cudarc?style=for-the-badge)](https://crates.io/crates/cudarc)
[![docs.rs](https://img.shields.io/docsrs/cudarc?label=docs.rs%20latest&style=for-the-badge)](https://docs.rs/cudarc)

Checkout cudarc on [crates.io](https://crates.io/crates/cudarc) and [docs.rs](https://docs.rs/cudarc/latest/cudarc/).

Safe abstractions over:
1. [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
2. [NVRTC API](https://docs.nvidia.com/cuda/nvrtc/index.html)
3. [cuRAND API](https://docs.nvidia.com/cuda/curand/index.html)
4. [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
5. [cuBLASLt API](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)

**Pre-alpha state**, expect breaking changes and not all cuda functions
contain a safe wrapper. **Contributions welcome for any that aren't included!**

# Design

Goals are:
1. As safe as possible (there will still be a lot of unsafe due to ffi & async)
2. As ergonomic as possible
3. Allow mixing of high level `safe` apis, with low level `sys` apis

To that end there are three levels to each wrapper (by default the safe api is exported):
```rust
use cudarc::driver::{safe, result, sys};
use cudarc::nvrtc::{safe, result, sys};
use cudarc::cublas::{safe, result, sys};
use cudarc::cublaslt::{safe, result, sys};
use cudarc::curand::{safe, result, sys};
```

where:
1. `sys` is the raw ffi apis generated with bindgen
2. `result` is a very small wrapper around sys to return `Result` from each function
3. `safe` is a wrapper around result/sys to provide safe abstractions

*Heavily recommend sticking with safe APIs*

# API Preview

It's easy to create a new device and transfer data to the gpu:

```rust
let dev = cudarc::driver::CudaDevice::new(0)?;

// allocate buffers
let inp = dev.htod_copy(vec![1.0f32; 100])?;
let mut out = dev.alloc_zeros::<f32>(100)?;
```

You can also use the nvrtc api to compile kernels at runtime:

```rust
let ptx = cudarc::nvrtc::compile_ptx("
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}")?;

// and dynamically load it into the device
dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;
```

`cudarc` provides a very simple interface to launch kernels, tuples
are the arguments!

```rust
let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
let cfg = LaunchConfig::for_num_elems(100);
unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;
```

And of course it's easy to copy things back to host after you're done:

```rust
let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
assert_eq!(out_host, [1.0; 100].map(f32::sin));
```

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
