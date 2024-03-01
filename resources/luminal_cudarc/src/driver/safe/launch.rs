use crate::driver::{result, sys};

use super::alloc::DeviceRepr;
use super::core::{CudaDevice, CudaFunction, CudaModule, CudaStream};

use std::{sync::Arc, vec::Vec};

impl CudaDevice {
    /// Whether a module and function are currently loaded into the device.
    pub fn has_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> bool {
        let modules = self.modules.read();
        #[cfg(not(feature = "no-std"))]
        let modules = modules.unwrap();

        modules
            .get(module_name)
            .map_or(false, |module| module.has_func(func_name))
    }

    /// Retrieves a [CudaFunction] that was registered under `module_name` and `func_name`.
    pub fn get_func(self: &Arc<Self>, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let modules = self.modules.read();
        #[cfg(not(feature = "no-std"))]
        let modules = modules.unwrap();

        modules
            .get(module_name)
            .and_then(|m| m.get_func(func_name))
            .map(|cu_function| CudaFunction {
                cu_function,
                device: self.clone(),
            })
    }
}

impl CudaModule {
    /// Returns reference to function with `name`. If function
    /// was not already loaded into CudaModule, then `None`
    /// is returned.
    pub(crate) fn get_func(&self, name: &str) -> Option<sys::CUfunction> {
        self.functions.get(name).cloned()
    }

    pub(crate) fn has_func(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

impl CudaFunction {
    #[inline(always)]
    unsafe fn launch_async_impl(
        self,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::DriverError> {
        self.device.bind_to_thread()?;
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.device.stream,
            params,
        )
    }

    #[inline(always)]
    unsafe fn par_launch_async_impl(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::DriverError> {
        self.device.bind_to_thread()?;
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.stream,
            params,
        )
    }
}

/// Configuration for [result::launch_kernel]
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
/// for description of each parameter.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n + 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Consumes a [CudaFunction] to execute asychronously on the device with
/// params determined by generic parameter `Params`.
///
/// This is impl'd multiple times for different number and types of params. In
/// general, `Params` should impl [DeviceRepr].
///
/// ```ignore
/// # use cudarc::driver::*;
/// # let dev = CudaDevice::new(0).unwrap();
/// let my_kernel: CudaFunction = dev.get_func("my_module", "my_kernel").unwrap();
/// let cfg: LaunchConfig = LaunchConfig {
///     grid_dim: (1, 1, 1),
///     block_dim: (1, 1, 1),
///     shared_mem_bytes: 0,
/// };
/// let params = (1i32, 2u64, 3usize);
/// unsafe { my_kernel.launch(cfg, params) }.unwrap();
/// ```
///
/// # Safety
///
/// This is not safe really ever, because there's no garuntee that `Params`
/// will work for any [CudaFunction] passed in. Great care should be taken
/// to ensure that [CudaFunction] works with `Params` and that the correct
/// parameters have `&mut` in front of them.
///
/// Additionally, kernels can mutate data that is marked as immutable,
/// such as `&CudaSlice<T>`.
///
/// See [LaunchAsync::launch] for more details
pub unsafe trait LaunchAsync<Params> {
    /// Launches the [CudaFunction] with the corresponding `Params`.
    ///
    /// # Safety
    ///
    /// This method is **very** unsafe.
    ///
    /// See cuda documentation notes on this as well:
    /// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#functions>
    ///
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point after the
    /// function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    /// in accessing/modifying values past memory limits.
    ///
    /// ## Asynchronous mutation
    ///
    /// Since this library queues kernels to be launched on a single
    /// stream, and really the only way to modify [crate::driver::CudaSlice] is through
    /// kernels, mutating the same [crate::driver::CudaSlice] with multiple kernels
    /// is safe. This is because each kernel is executed sequentially
    /// on the stream.
    ///
    /// **Modifying a value on the host that is in used by a
    /// kernel is undefined behavior.** But is hard to do
    /// accidentally.
    ///
    /// Also for this reason, do not pass in any values to kernels
    /// that can be modified on the host. This is the reason
    /// [DeviceRepr] is not implemented for rust primitive
    /// references.
    ///
    /// ## Use after free
    ///
    /// Since the drop implementation for [crate::driver::CudaSlice] also occurs
    /// on the device's single stream, any kernels launched before
    /// the drop will complete before the value is actually freed.
    ///
    /// **If you launch a kernel or drop a value on a different stream
    /// this may not hold**
    unsafe fn launch(self, cfg: LaunchConfig, params: Params) -> Result<(), result::DriverError>;

    /// Launch the function on a stream concurrent to the device's default
    /// work stream.
    ///
    /// # Safety
    /// This method is even more unsafe than [LaunchAsync::launch], all the same rules apply,
    /// except now things are executing in parallel to each other.
    ///
    /// That means that if any of the kernels modify the same memory location, you'll get race
    /// conditions or potentially undefined behavior.
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), result::DriverError>;
}

unsafe impl LaunchAsync<&mut [*mut std::ffi::c_void]> for CudaFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::DriverError> {
        self.launch_async_impl(cfg, args)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), result::DriverError> {
        self.par_launch_async_impl(stream, cfg, args)
    }
}

unsafe impl LaunchAsync<&mut Vec<*mut std::ffi::c_void>> for CudaFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        args: &mut Vec<*mut std::ffi::c_void>,
    ) -> Result<(), result::DriverError> {
        self.launch_async_impl(cfg, args)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: &mut Vec<*mut std::ffi::c_void>,
    ) -> Result<(), result::DriverError> {
        self.par_launch_async_impl(stream, cfg, args)
    }
}

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
unsafe impl<$($Vars: DeviceRepr),*> LaunchAsync<($($Vars, )*)> for CudaFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), result::DriverError> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.launch_async_impl(cfg, params)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        args: ($($Vars, )*)
    ) -> Result<(), result::DriverError> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.par_launch_async_impl(stream, cfg, params)
    }
}
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{
        driver::{DeviceSlice, DriverError},
        nvrtc::compile_ptx_with_opts,
    };

    use super::*;

    #[test]
    fn test_mut_into_kernel_param_no_inc_rc() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    #[test]
    fn test_ref_into_kernel_param_inc_rc() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 1].to_vec()).unwrap();
        let _r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        let _ = (&t).as_kernel_param();
        assert_eq!(Arc::strong_count(&device), 3);
    }

    const SIN_CU: &str = "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}";

    #[test]
    fn test_launch_with_mut_and_ref_cudarc() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();

        let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();

        let mut b_dev = a_dev.clone();

        unsafe {
            sin_kernel.launch(
                LaunchConfig::for_num_elems(10),
                (&mut b_dev, &a_dev, 10usize),
            )
        }
        .unwrap();

        let b_host = dev.sync_reclaim(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    #[test]
    fn test_large_launches() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();
        for numel in [256, 512, 1024, 1280, 1536, 2048] {
            let mut a = Vec::with_capacity(numel);
            a.resize(numel, 1.0f32);

            let a = dev.htod_copy(a).unwrap();
            let mut b = dev.alloc_zeros::<f32>(numel).unwrap();

            let sin_kernel = dev.get_func("sin", "sin_kernel").unwrap();
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            unsafe { sin_kernel.launch(cfg, (&mut b, &a, numel)) }.unwrap();

            let b = dev.sync_reclaim(b).unwrap();
            for v in b {
                assert_eq!(v, 0.841471);
            }
        }
    }

    #[test]
    fn test_launch_with_views() {
        let ptx = compile_ptx_with_opts(SIN_CU, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();

        let a_host = [-1.0f32, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];
        let a_dev = dev.htod_copy(a_host.clone().to_vec()).unwrap();
        let mut b_dev = a_dev.clone();

        for i in 0..5 {
            let a_sub = a_dev.try_slice(i * 2..).unwrap();
            assert_eq!(a_sub.len, 10 - 2 * i);
            let mut b_sub = b_dev.try_slice_mut(i * 2..).unwrap();
            assert_eq!(b_sub.len, 10 - 2 * i);
            let f = dev.get_func("sin", "sin_kernel").unwrap();
            unsafe { f.launch(LaunchConfig::for_num_elems(2), (&mut b_sub, &a_sub, 2usize)) }
                .unwrap();
        }

        let b_host = dev.sync_reclaim(b_dev).unwrap();

        for (a_i, b_i) in a_host.iter().zip(b_host.iter()) {
            let expected = a_i.sin();
            assert!((b_i - expected).abs() <= 1e-6);
        }

        drop(a_dev);
    }

    const TEST_KERNELS: &str = "
extern \"C\" __global__ void int_8bit(signed char s_min, char s_max, unsigned char u_min, unsigned char u_max) {
    assert(s_min == -128);
    assert(s_max == 127);
    assert(u_min == 0);
    assert(u_max == 255);
}

extern \"C\" __global__ void int_16bit(signed short s_min, short s_max, unsigned short u_min, unsigned short u_max) {
    assert(s_min == -32768);
    assert(s_max == 32767);
    assert(u_min == 0);
    assert(u_max == 65535);
}

extern \"C\" __global__ void int_32bit(signed int s_min, int s_max, unsigned int u_min, unsigned int u_max) {
    assert(s_min == -2147483648);
    assert(s_max == 2147483647);
    assert(u_min == 0);
    assert(u_max == 4294967295);
}

extern \"C\" __global__ void int_64bit(signed long s_min, long s_max, unsigned long u_min, unsigned long u_max) {
    assert(s_min == -9223372036854775808);
    assert(s_max == 9223372036854775807);
    assert(u_min == 0);
    assert(u_max == 18446744073709551615);
}

extern \"C\" __global__ void floating(float f, double d) {
    assert(fabs(f - 1.2345678) <= 1e-7);
    assert(fabs(d - -10.123456789876543) <= 1e-16);
}
";

    #[test]
    fn test_launch_with_8bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_8bit"]).unwrap();
        let f = dev.get_func("tests", "int_8bit").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (i8::MIN, i8::MAX, u8::MIN, u8::MAX),
            )
        }
        .unwrap();

        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_16bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_16bit"]).unwrap();
        let f = dev.get_func("tests", "int_16bit").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (i16::MIN, i16::MAX, u16::MIN, u16::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_32bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_32bit"]).unwrap();
        let f = dev.get_func("tests", "int_32bit").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (i32::MIN, i32::MAX, u32::MIN, u32::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_64bit() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["int_64bit"]).unwrap();
        let f = dev.get_func("tests", "int_64bit").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (i64::MIN, i64::MAX, u64::MIN, u64::MAX),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_launch_with_floats() {
        let ptx = compile_ptx_with_opts(TEST_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["floating"]).unwrap();
        let f = dev.get_func("tests", "floating").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (1.2345678f32, -10.123456789876543f64),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    #[cfg(feature = "f16")]
    const HALF_KERNELS: &str = "
#include \"cuda_fp16.h\"

extern \"C\" __global__ void halfs(__half h) {
    assert(__habs(h - __float2half(1.234)) <= __float2half(1e-4));
}
";

    #[cfg(feature = "f16")]
    #[test]
    fn test_launch_with_half() {
        use crate::nvrtc::CompileOptions;

        let ptx = compile_ptx_with_opts(
            HALF_KERNELS,
            CompileOptions {
                include_paths: std::vec!["/usr/include".into()],
                arch: Some("compute_53"),
                ..Default::default()
            },
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["halfs"]).unwrap();
        let f = dev.get_func("tests", "halfs").unwrap();
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1),
                (half::f16::from_f32(1.234),),
            )
        }
        .unwrap();
        dev.synchronize().unwrap();
    }

    const SLOW_KERNELS: &str = "
extern \"C\" __global__ void slow_worker(const float *data, const size_t len, float *out) {
    float tmp = 0.0;
    for(size_t i = 0; i < 1000000; i++) {
        tmp += data[i % len];
    }
    *out = tmp;
}
";

    #[test]
    fn test_par_launch() -> Result<(), DriverError> {
        let ptx = compile_ptx_with_opts(SLOW_KERNELS, Default::default()).unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "tests", &["slow_worker"]).unwrap();
        let slice = dev.alloc_zeros::<f32>(1000)?;
        let mut a = dev.alloc_zeros::<f32>(1)?;
        let mut b = dev.alloc_zeros::<f32>(1)?;
        let cfg = LaunchConfig::for_num_elems(1);

        let start = Instant::now();
        {
            // launch two kernels on the default stream
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch(cfg, (&slice, slice.len(), &mut a))? };
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch(cfg, (&slice, slice.len(), &mut b))? };
            dev.synchronize()?;
        }
        let double_launch_s = start.elapsed().as_secs_f64();

        let start = Instant::now();
        {
            // create a new stream & launch them concurrently
            let stream = dev.fork_default_stream()?;
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch(cfg, (&slice, slice.len(), &mut a))? };
            let f = dev.get_func("tests", "slow_worker").unwrap();
            unsafe { f.launch_on_stream(&stream, cfg, (&slice, slice.len(), &mut b))? };
            dev.wait_for(&stream)?;
            dev.synchronize()?;
        }
        let par_launch_s = start.elapsed().as_secs_f64();

        assert!(
            (double_launch_s - 2.0 * par_launch_s).abs() < 20.0 / 100.0,
            "par={:?} dbl={:?}",
            par_launch_s,
            double_launch_s
        );
        Ok(())
    }
}
