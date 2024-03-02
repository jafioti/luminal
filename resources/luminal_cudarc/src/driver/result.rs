//! A thin wrapper around [sys].
//!
//! While all the functions here will return [Result], they are
//! mostly all still unsafe because order of operations
//! really matters.
//!
//! This also only exposes the `*_async` version of functions
//! because mixing the two is confusing and even more unsafe.
//!
//! This module also groups functions into sub-modules
//! to make naming easier. For example [sys::cuStreamCreate()]
//! turns into [stream::create()], where [stream] is a module.

use super::sys;
use core::ffi::{c_uchar, c_uint, c_void, CStr};
use std::mem::MaybeUninit;

/// Wrapper around [sys::CUresult]. See
/// nvidia's [CUresult docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub sys::CUresult);

impl sys::CUresult {
    #[inline]
    pub fn result(self) -> Result<(), DriverError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

impl DriverError {
    /// Gets the name for this error.
    ///
    /// See [cuGetErrorName() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468)
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Gets the error string for this error.
    ///
    /// See [cuGetErrorString() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada)
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string().unwrap();
        f.debug_tuple("DriverError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DriverError {}

/// Initializes the CUDA driver API.
/// **MUST BE CALLED BEFORE ANYTHING ELSE**
///
/// See [cuInit() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3)
pub fn init() -> Result<(), DriverError> {
    unsafe { sys::cuInit(0).result() }
}

pub mod device {
    //! Device management functions (`cuDevice*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

    use super::{sys, DriverError};
    use core::ffi::c_int;
    use std::mem::MaybeUninit;

    /// Get a device for a specific ordinal.
    /// See [cuDeviceGet() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb).
    pub fn get(ordinal: c_int) -> Result<sys::CUdevice, DriverError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// Gets the number of available devices.
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74)
    pub fn get_count() -> Result<c_int, DriverError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total amount of memory in bytes on the device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub unsafe fn total_mem(dev: sys::CUdevice) -> Result<usize, DriverError> {
        let mut bytes = MaybeUninit::uninit();
        sys::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
        Ok(bytes.assume_init())
    }

    /// Get an attribute of a device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8c6e2c7b5c7c8b7e6f7f4c2b7f6d9c5d)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub unsafe fn get_attribute(
        dev: sys::CUdevice,
        attrib: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = MaybeUninit::uninit();
        sys::cuDeviceGetAttribute(value.as_mut_ptr(), attrib, dev).result()?;
        Ok(value.assume_init())
    }
}

pub mod function {
    use super::sys::{self, CUfunction_attribute_enum};

    /// Sets the specific attribute of a cuda function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn set_function_attribute(
        f: sys::CUfunction,
        attribute: CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), super::DriverError> {
        unsafe {
            sys::cuFuncSetAttribute(f, attribute, value).result()?;
        }

        Ok(())
    }
}

pub mod occupancy {

    use core::{
        ffi::{c_int, c_uint},
        mem::MaybeUninit,
    };

    use super::{sys, DriverError};

    /// Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gae02af6a9df9e1bbd51941af631bce69)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn available_dynamic_shared_mem_per_block(
        f: sys::CUfunction,
        num_blocks: c_int,
        block_size: c_int,
    ) -> Result<usize, DriverError> {
        let mut dynamic_smem_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyAvailableDynamicSMemPerBlock(
                dynamic_smem_size.as_mut_ptr(),
                f,
                num_blocks,
                block_size,
            )
            .result()?;
        }
        Ok(dynamic_smem_size.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn max_active_block_per_multiprocessor(
        f: sys::CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> Result<i32, DriverError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g8f1da4d4983e5c3025447665423ae2c2)
    ///
    /// # Safety
    /// Function must exist. No invalid flags.
    pub unsafe fn max_active_block_per_multiprocessor_with_flags(
        f: sys::CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
        flags: c_uint,
    ) -> Result<i32, DriverError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
                flags,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    pub unsafe fn max_potential_block_size(
        f: sys::CUfunction,
        block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
    ) -> Result<(i32, i32), DriverError> {
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSize(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                f,
                block_size_to_dynamic_smem_size,
                dynamic_smem_size,
                block_size_limit,
            )
            .result()?;
        }
        Ok((min_grid_size.assume_init(), block_size.assume_init()))
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    pub unsafe fn max_potential_block_size_with_flags(
        f: sys::CUfunction,
        block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
        flags: c_uint,
    ) -> Result<(i32, i32), DriverError> {
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSizeWithFlags(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                f,
                block_size_to_dynamic_smem_size,
                dynamic_smem_size,
                block_size_limit,
                flags,
            )
            .result()?;
        }
        Ok((min_grid_size.assume_init(), block_size.assume_init()))
    }
}

pub mod primary_ctx {
    //! Primary context management functions (`cuDevicePrimaryCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

    use super::{sys, DriverError};
    use std::mem::MaybeUninit;

    /// Creates a primary context on the device and pushes it onto the primary context stack.
    /// Call [release] to free it.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300)
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn retain(dev: sys::CUdevice) -> Result<sys::CUcontext, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// Release a reference to the current primary context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad).
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn release(dev: sys::CUdevice) -> Result<(), DriverError> {
        sys::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

pub mod ctx {
    //! Context management functions (`cuCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

    use super::{sys, DriverError};
    use std::mem::MaybeUninit;

    /// Binds the specified CUDA context to the calling CPU thread.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7)
    ///
    /// # Safety
    ///
    /// This has weird behavior depending on the value of `ctx`. See cuda docs for more info.
    /// In general this should only be called with an already initialized context,
    /// and one that wasn't already freed.
    pub unsafe fn set_current(ctx: sys::CUcontext) -> Result<(), DriverError> {
        sys::cuCtxSetCurrent(ctx).result()
    }

    /// Returns the CUDA context bound to the calling CPU thread if there is one.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0)
    pub fn get_current() -> Result<Option<sys::CUcontext>, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        unsafe {
            sys::cuCtxGetCurrent(ctx.as_mut_ptr()).result()?;
            let ctx: sys::CUcontext = ctx.assume_init();
            if ctx.is_null() {
                Ok(None)
            } else {
                Ok(Some(ctx))
            }
        }
    }
}

pub mod stream {
    //! Stream management functions (`cuStream*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM).

    use super::{sys, DriverError};
    use std::mem::MaybeUninit;

    /// The kind of stream to initialize.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub enum StreamKind {
        /// From cuda docs:
        /// > Default stream creation flag.
        Default,

        /// From cuda docs:
        /// > Specifies that work running in the created stream
        /// > may run concurrently with work in stream 0 (the NULL stream),
        /// > and that the created stream should perform no implicit
        /// > synchronization with stream 0.
        NonBlocking,
    }

    impl StreamKind {
        fn flags(self) -> sys::CUstream_flags {
            match self {
                Self::Default => sys::CUstream_flags::CU_STREAM_DEFAULT,
                Self::NonBlocking => sys::CUstream_flags::CU_STREAM_NON_BLOCKING,
            }
        }
    }

    /// The null stream, which is just a null pointer. **Recommend not using this.**
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)
    pub fn null() -> sys::CUstream {
        std::ptr::null_mut()
    }

    /// Creates a stream with the specified kind.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub fn create(kind: StreamKind) -> Result<sys::CUstream, DriverError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn synchronize(stream: sys::CUstream) -> Result<(), DriverError> {
        sys::cuStreamSynchronize(stream).result()
    }

    /// Destroys a stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn destroy(stream: sys::CUstream) -> Result<(), DriverError> {
        sys::cuStreamDestroy_v2(stream).result()
    }

    /// Make a compute stream wait on an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f)
    ///
    /// # Safety
    /// 1. Both stream and event must not have been freed already
    pub unsafe fn wait_event(
        stream: sys::CUstream,
        event: sys::CUevent,
        flags: sys::CUevent_wait_flags,
    ) -> Result<(), DriverError> {
        sys::cuStreamWaitEvent(stream, event, flags as u32).result()
    }
}

/// Allocates memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory return by this is unset, which may be invalid for `T`.
/// 3. All uses of this memory must be on the same stream.
pub unsafe fn malloc_async(
    stream: sys::CUstream,
    num_bytes: usize,
) -> Result<sys::CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
    Ok(dev_ptr.assume_init())
}

/// Allocates memory
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467)
///
/// # Safety
/// 1. The memory return by this is unset, which may be invalid for `T`.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
    Ok(dev_ptr.assume_init())
}

/// Frees memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory should have been allocated on this stream.
/// 3. The memory should not have been freed already (double free)
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    sys::cuMemFreeAsync(dptr, stream).result()
}

/// Allocates memory
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a)
///
/// # Safety
/// 1. The memory should have been allocated with malloc_sync
pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemFree_v2(dptr).result()
}

/// Frees device memory.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a)
///
/// # Safety
/// 1. Memory must only be freed once.
/// 2. All async accesses to this pointer must have been completed.
pub unsafe fn memory_free(device_ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemFree_v2(device_ptr).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memset_d8_async(
    dptr: sys::CUdeviceptr,
    uc: c_uchar,
    num_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemsetD8Async(dptr, uc, num_bytes, stream).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memset_d8_sync(
    dptr: sys::CUdeviceptr,
    uc: c_uchar,
    num_bytes: usize,
) -> Result<(), DriverError> {
    sys::cuMemsetD8_v2(dptr, uc, num_bytes).result()
}

/// Copies memory from Host to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169)
///
/// # Safety
/// **This function is asynchronous** in most cases, so the data from `src`
/// will be copied at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
/// 4. `src` must not be moved
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: &[T],
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyHtoDAsync_v2(
        dst,
        src.as_ptr() as *const _,
        std::mem::size_of_val(src),
        stream,
    )
    .result()
}

/// Copies memory from Host to Device
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169)
///
/// # Safety
/// **This function is synchronous**///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. `src` must not be moved
pub unsafe fn memcpy_htod_sync<T>(dst: sys::CUdeviceptr, src: &[T]) -> Result<(), DriverError> {
    sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const _, std::mem::size_of_val(src)).result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362)
///
/// # Safety
/// **This function is asynchronous** in most cases, so `dst` will be
/// mutated at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut [T],
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoHAsync_v2(
        dst.as_mut_ptr() as *mut _,
        src,
        std::mem::size_of_val(dst),
        stream,
    )
    .result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893)
///
/// # Safety
/// **This function is synchronous**
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memcpy_dtoh_sync<T>(dst: &mut [T], src: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, std::mem::size_of_val(dst)).result()
}

/// Copies memory from Device to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtod_async(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream).result()
}

/// Copies memory from Device to Device
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should not have been freed already (double free)
pub unsafe fn memcpy_dtod_sync(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_bytes: usize,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoD_v2(dst, src, num_bytes).result()
}

/// Returns (free, total) memory in bytes.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0)
pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
    let mut free = 0;
    let mut total = 0;
    unsafe { sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) }.result()?;
    Ok((free, total))
}

pub mod module {
    //! Module management functions (`cuModule*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

    use super::{sys, DriverError};
    use core::ffi::c_void;
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    /// Loads a compute module from a given file.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3)
    pub fn load(fname: CString) -> Result<sys::CUmodule, DriverError> {
        let fname_ptr = fname.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            sys::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// Load a module's data:
    ///
    /// > The pointer may be obtained by mapping a cubin or PTX or fatbin file,
    /// > passing a cubin or PTX or fatbin file as a NULL-terminated text string,
    /// > or incorporating a cubin or fatbin object into the executable resources
    /// > and using operating system calls such as Windows FindResource() to obtain the pointer.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b)
    ///
    /// # Safety
    /// The image must be properly formed pointer
    pub unsafe fn load_data(image: *const c_void) -> Result<sys::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        sys::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// Returns a function handle from the given module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf)
    ///
    /// # Safety
    /// `module` must be a properly allocated and not freed module.
    pub unsafe fn get_function(
        module: sys::CUmodule,
        name: CString,
    ) -> Result<sys::CUfunction, DriverError> {
        let name_ptr = name.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        sys::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr).result()?;
        Ok(func.assume_init())
    }

    /// Unloads a module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b)
    ///
    /// # Safety
    /// `module` must not have be unloaded already.
    pub unsafe fn unload(module: sys::CUmodule) -> Result<(), DriverError> {
        sys::cuModuleUnload(module).result()
    }
}

pub mod event {
    use super::{sys, DriverError};
    use std::mem::MaybeUninit;

    /// Creates an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db)
    pub fn create(flags: sys::CUevent_flags) -> Result<sys::CUevent, DriverError> {
        let mut event = MaybeUninit::uninit();
        unsafe {
            sys::cuEventCreate(event.as_mut_ptr(), flags as u32).result()?;
            Ok(event.assume_init())
        }
    }

    /// Records an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1)
    ///
    /// # Safety
    /// This function is unsafe because event can be a null event, in which case
    pub unsafe fn record(event: sys::CUevent, stream: sys::CUstream) -> Result<(), DriverError> {
        unsafe { sys::cuEventRecord(event, stream).result() }
    }

    /// Computes the elapsed time (in milliseconds) between two events.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97)
    /// # Safety
    /// 1. Events must have been created by [create]
    /// 2. They should be on the same stream
    /// 3. They must not have been destroyed.
    pub unsafe fn elapsed(start: sys::CUevent, end: sys::CUevent) -> Result<f32, DriverError> {
        let mut ms: f32 = 0.0;
        unsafe {
            sys::cuEventElapsedTime((&mut ms) as *mut _, start, end).result()?;
        }
        Ok(ms)
    }

    /// Destroys an event.
    ///
    /// > An event may be destroyed before it is complete (i.e., while cuEventQuery() would return CUDA_ERROR_NOT_READY).
    /// > In this case, the call does not block on completion of the event,
    /// > and any associated resources will automatically be released asynchronously at completion.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef)
    ///
    /// # Safety
    /// 1. Event must not have been freed already
    pub unsafe fn destroy(event: sys::CUevent) -> Result<(), DriverError> {
        sys::cuEventDestroy_v2(event).result()
    }
}

/// Launches a cuda functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
///
/// # Safety
/// This method is **very unsafe**.
///
/// 1. The cuda function must be a valid handle returned from a non-unloaded module.
/// 2. This is asynchronous, so the results of calling this function happen
/// at a later point after this function returns.
/// 3. All parameters used for this kernel should have been allocated by stream (I think?)
/// 4. The cuda kernel has mutable access to every parameter, that means every parameter
/// can change at a later point after callign this function. *Even non-mutable references*.
#[inline]
pub unsafe fn launch_kernel(
    f: sys::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    sys::cuLaunchKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    )
    .result()
}

pub mod external_memory {
    use std::mem::MaybeUninit;

    use super::{sys, DriverError};

    /// Imports an external memory object, in this case an OpaqueFd.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(unix)]
    pub unsafe fn import_external_memory_opaque_fd(
        fd: std::os::fd::RawFd,
        size: u64,
    ) -> Result<sys::CUexternalMemory, DriverError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd },
            size,
            ..Default::default()
        };
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description).result()?;
        Ok(external_memory.assume_init())
    }

    /// Imports an external memory object, in this case an OpaqueWin32 handle.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(windows)]
    pub unsafe fn import_external_memory_opaque_win32(
        handle: std::os::windows::io::RawHandle,
        size: u64,
    ) -> Result<sys::CUexternalMemory, DriverError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
                win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    handle,
                    name: std::ptr::null(),
                },
            },
            size,
            ..Default::default()
        };
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description).result()?;
        Ok(external_memory.assume_init())
    }

    /// Destroys an external memory object.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052)
    ///
    /// # Safety
    /// 1. Any mapped buffers onto this object must already be freed.
    /// 2. The external memory must only be destroyed once.
    pub unsafe fn destroy_external_memory(
        external_memory: sys::CUexternalMemory,
    ) -> Result<(), DriverError> {
        sys::cuDestroyExternalMemory(external_memory).result()
    }

    /// Maps a buffer onto an imported memory object.
    ///
    /// The buffer must be freed using [`memory_free`](super::memory_free).
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91)
    ///
    /// # Safety
    /// Mapped buffers may overlap.
    pub unsafe fn get_mapped_buffer(
        external_memory: sys::CUexternalMemory,
        offset: u64,
        size: u64,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut device_ptr = MaybeUninit::uninit();
        let buffer_description = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset,
            size,
            ..Default::default()
        };
        sys::cuExternalMemoryGetMappedBuffer(
            device_ptr.as_mut_ptr(),
            external_memory,
            &buffer_description,
        )
        .result()?;
        Ok(device_ptr.assume_init())
    }
}
