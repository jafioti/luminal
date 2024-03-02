use crate::driver::{result, sys};

use super::core::{CudaDevice, CudaSlice, CudaView, CudaViewMut};
use super::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};

use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};

/// Something that can be copied to device memory and
/// turned into a parameter for [result::launch_kernel].
///
/// # Safety
///
/// This is unsafe because a struct should likely
/// be `#[repr(C)]` to be represented in cuda memory,
/// and not all types are valid.
pub unsafe trait DeviceRepr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl DeviceRepr for half::bf16 {}

unsafe impl<T: DeviceRepr> DeviceRepr for &mut CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<T: DeviceRepr> DeviceRepr for &CudaSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.cu_device_ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceRepr> DeviceRepr for &CudaView<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceRepr> DeviceRepr for &mut CudaViewMut<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl<T> CudaSlice<T> {
    /// Takes ownership of the underlying [sys::CUdeviceptr]. **It is up
    /// to the owner to free this value**.
    ///
    /// Drops the underlying host_buf if there is one.
    pub fn leak(mut self) -> sys::CUdeviceptr {
        if let Some(host_buf) = std::mem::take(&mut self.host_buf) {
            drop(host_buf);
        }
        let ptr = self.cu_device_ptr;
        std::mem::forget(self);
        ptr
    }
}

impl CudaDevice {
    /// Creates a [CudaSlice] from a [sys::CUdeviceptr]. Useful in conjunction with
    /// [`CudaSlice::leak()`].
    ///
    /// # Safety
    /// - `cu_device_ptr` must be a valid allocation
    /// - `cu_device_ptr` must space for `len * std::mem::size_of<T>()` bytes
    /// - The memory may not be valid for type `T`, so some sort of memset operation
    ///   should be called on the memory.
    pub unsafe fn upgrade_device_ptr<T>(
        self: &Arc<Self>,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        CudaSlice {
            cu_device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        }
    }
}

impl CudaDevice {
    /// Allocates an empty [CudaSlice] with 0 length.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, result::DriverError> {
        self.bind_to_thread()?;
        let cu_device_ptr = unsafe {
            if self.is_async {
                result::malloc_async(self.stream, 0)?
            } else {
                result::malloc_sync(0)?
            }
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len: 0,
            device: self.clone(),
            host_buf: None,
        })
    }

    /// Allocates device memory and increments the reference counter of [CudaDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        self.bind_to_thread()?;
        let cu_device_ptr = if self.is_async {
            result::malloc_async(self.stream, len * std::mem::size_of::<T>())?
        } else {
            result::malloc_sync(len * std::mem::size_of::<T>())?
        };
        Ok(CudaSlice {
            cu_device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        })
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros<T: ValidAsZeroBits + DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut dst)?;
        Ok(dst)
    }

    /// Sets all memory to 0 asynchronously.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn memset_zeros<T: ValidAsZeroBits + DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        self.bind_to_thread()?;
        if self.is_async {
            unsafe {
                result::memset_d8_async(*dst.device_ptr_mut(), 0, dst.num_bytes(), self.stream)
            }
        } else {
            unsafe { result::memset_d8_sync(*dst.device_ptr_mut(), 0, dst.num_bytes()) }
        }
    }

    /// Device to device copy (safe version of [result::memcpy_dtod_async]).
    ///
    /// # Panics
    ///
    /// If the length of the two values are different
    ///
    /// # Safety
    /// 1. We are guarunteed that `src` and `dst` are pointers to the same underlying
    ///     type `T`
    /// 2. Since they are both references, they can't have been freed
    /// 3. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtod_copy<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        self.bind_to_thread()?;
        if self.is_async {
            unsafe {
                result::memcpy_dtod_async(
                    *dst.device_ptr_mut(),
                    *src.device_ptr(),
                    src.len() * std::mem::size_of::<T>(),
                    self.stream,
                )
            }
        } else {
            unsafe {
                result::memcpy_dtod_sync(
                    *dst.device_ptr_mut(),
                    *src.device_ptr(),
                    src.len() * std::mem::size_of::<T>(),
                )
            }
        }
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy<T: Unpin + DeviceRepr>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.htod_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy_into<T: DeviceRepr + Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        dst.host_buf = Some(Pin::new(src));
        self.bind_to_thread()?;
        if self.is_async {
            unsafe {
                result::memcpy_htod_async(
                    dst.cu_device_ptr,
                    dst.host_buf.as_ref().unwrap(),
                    self.stream,
                )
            }?
        } else {
            unsafe { result::memcpy_htod_sync(dst.cu_device_ptr, dst.host_buf.as_ref().unwrap()) }?
        }
        Ok(())
    }

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::htod_copy()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &[T],
    ) -> Result<CudaSlice<T>, result::DriverError> {
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.htod_sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [CudaDevice::htod_copy()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy_into<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        self.bind_to_thread()?;
        if self.is_async {
            unsafe { result::memcpy_htod_async(*dst.device_ptr_mut(), src, self.stream) }?;
        } else {
            unsafe { result::memcpy_htod_sync(*dst.device_ptr_mut(), src) }?;
        }
        self.synchronize()
    }

    /// Synchronously copies device memory into host memory.
    /// Unlike [`CudaDevice::dtoh_sync_copy_into`] this returns a [`Vec<T>`].
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` (after returning) it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    #[allow(clippy::uninit_vec)]
    pub fn dtoh_sync_copy<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, result::DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        unsafe { dst.set_len(src.len()) };
        self.dtoh_sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies device memory into host memory
    ///
    /// Use [`CudaDevice::dtoh_sync_copy`] if you need [`Vec<T>`] and can't provide
    /// a correctly sized slice.
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtoh_sync_copy_into<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut [T],
    ) -> Result<(), result::DriverError> {
        assert_eq!(src.len(), dst.len());
        self.bind_to_thread()?;
        if self.is_async {
            unsafe { result::memcpy_dtoh_async(dst, *src.device_ptr(), self.stream) }?;
        } else {
            unsafe { result::memcpy_dtoh_sync(dst, *src.device_ptr()) }?;
        }
        self.synchronize()
    }

    /// Synchronously de-allocates `src` and converts it into it's host value.
    /// You can just [drop] the slice if you don't need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_reclaim<T: Clone + Default + DeviceRepr + Unpin>(
        self: &Arc<Self>,
        mut src: CudaSlice<T>,
    ) -> Result<Vec<T>, result::DriverError> {
        let buf = src.host_buf.take();
        let mut buf = buf.unwrap_or_else(|| {
            let mut b = Vec::with_capacity(src.len);
            b.resize(src.len, Default::default());
            Pin::new(b)
        });
        self.dtoh_sync_copy_into(&src, &mut buf)?;
        Ok(Pin::into_inner(buf))
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> Result<(), result::DriverError> {
        self.bind_to_thread()?;
        unsafe { result::stream::synchronize(self.stream) }
    }
}

/// Marker trait to indicate that the type is valid
/// when all of its bits are set to 0.
///
/// # Safety
/// Not all types are valid when all bits are set to 0.
/// Be very sure when implementing this trait!
pub unsafe trait ValidAsZeroBits {}
unsafe impl ValidAsZeroBits for bool {}
unsafe impl ValidAsZeroBits for i8 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for i128 {}
unsafe impl ValidAsZeroBits for isize {}
unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for u64 {}
unsafe impl ValidAsZeroBits for u128 {}
unsafe impl ValidAsZeroBits for usize {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::f16 {}
#[cfg(feature = "f16")]
unsafe impl ValidAsZeroBits for half::bf16 {}
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}
/// Implement `ValidAsZeroBits` for tuples if all elements are `ValidAsZeroBits`,
///
/// # Note
/// This will also implement `ValidAsZeroBits` for a tuple with one element
macro_rules! impl_tuples {
    ($t:tt) => {
        impl_tuples!(@ $t);
    };
    // the $l is in front of the reptition to prevent parsing ambiguities
    ($l:tt $(,$t:tt)+) => {
        impl_tuples!($($t),+);
        impl_tuples!(@ $l $(,$t)+);
    };
    (@ $($t:tt),+) => {
        unsafe impl<$($t: ValidAsZeroBits,)+> ValidAsZeroBits for ($($t,)+) {}
    };
}
impl_tuples!(A, B, C, D, E, F, G, H, I, J, K, L);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_build_arc_count() {
        let device = CudaDevice::new(0).unwrap();
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.alloc_zeros::<f32>(1).unwrap();
        assert!(t.host_buf.is_none());
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    fn test_post_take_arc_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        assert!(t.host_buf.is_some());
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([0.0f64; 10].to_vec()).unwrap();
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_arc_slice_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = Arc::new(device.htod_copy::<f64>([0.0; 10].to_vec()).unwrap());
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_release_counts() {
        let device = CudaDevice::new(0).unwrap();
        let t = device.htod_copy([1.0f32, 2.0, 3.0].to_vec()).unwrap();
        #[allow(clippy::redundant_clone)]
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 3);

        let r_host = device.sync_reclaim(r).unwrap();
        assert_eq!(&r_host, &[1.0, 2.0, 3.0]);
        assert_eq!(Arc::strong_count(&device), 2);

        drop(r_host);
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let device = CudaDevice::new(0).unwrap();
        let (free1, total1) = result::mem_get_info().unwrap();

        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        let (free2, total2) = result::mem_get_info().unwrap();
        assert_eq!(total1, total2);
        assert!(free2 < free1);

        drop(t);
        device.synchronize().unwrap();

        let (free3, total3) = result::mem_get_info().unwrap();
        assert_eq!(total2, total3);
        assert!(free3 > free2);
        assert_eq!(free3, free1);
    }

    #[test]
    fn test_device_copy_to_views() {
        let dev = CudaDevice::new(0).unwrap();

        let smalls = [
            dev.htod_copy(std::vec![-1.0f32, -0.8]).unwrap(),
            dev.htod_copy(std::vec![-0.6, -0.4]).unwrap(),
            dev.htod_copy(std::vec![-0.2, 0.0]).unwrap(),
            dev.htod_copy(std::vec![0.2, 0.4]).unwrap(),
            dev.htod_copy(std::vec![0.6, 0.8]).unwrap(),
        ];
        let mut big = dev.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.try_slice_mut(offset..offset + small.len()).unwrap();
            dev.dtod_copy(small, &mut sub).unwrap();
            offset += small.len();
        }

        assert_eq!(
            dev.sync_reclaim(big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_leak_and_upgrade() {
        let dev = CudaDevice::new(0).unwrap();

        let a = dev
            .htod_copy(std::vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
            .unwrap();

        let ptr = a.leak();
        let b = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 3) };
        assert_eq!(dev.dtoh_sync_copy(&b).unwrap(), &[1.0, 2.0, 3.0]);

        let ptr = b.leak();
        let c = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 5) };
        assert_eq!(dev.dtoh_sync_copy(&c).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// See https://github.com/coreylowman/cudarc/issues/160
    #[test]
    fn test_slice_is_freed_with_correct_context() {
        if CudaDevice::count().unwrap() < 2 {
            return;
        }
        let dev0 = CudaDevice::new(0).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let dev1 = CudaDevice::new(1).unwrap();
        drop(dev1);
        drop(dev0);
        drop(slice);
    }

    /// See https://github.com/coreylowman/cudarc/issues/161
    #[test]
    fn test_copy_uses_correct_context() {
        if CudaDevice::count().unwrap() < 2 {
            return;
        }
        let dev0 = CudaDevice::new(0).unwrap();
        let _dev1 = CudaDevice::new(1).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let _out = dev0.dtoh_sync_copy(&slice).unwrap();
    }
}
