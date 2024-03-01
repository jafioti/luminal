use core::mem::ManuallyDrop;
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use super::{CudaDevice, DevicePtr, DeviceSlice};
use crate::driver::{result, sys, DriverError};

impl CudaDevice {
    /// Import external memory from a [`File`].
    ///
    /// # Safety
    /// `size` must be the size of the external memory in bytes.
    #[cfg(any(unix, windows))]
    pub unsafe fn import_external_memory(
        self: &Arc<Self>,
        file: File,
        size: u64,
    ) -> Result<ExternalMemory, DriverError> {
        self.bind_to_thread()?;

        #[cfg(unix)]
        let external_memory = unsafe {
            use std::os::fd::AsRawFd;
            result::external_memory::import_external_memory_opaque_fd(file.as_raw_fd(), size)
        }?;
        #[cfg(windows)]
        let external_memory = unsafe {
            use std::os::windows::io::AsRawHandle;
            result::external_memory::import_external_memory_opaque_win32(file.as_raw_handle(), size)
        }?;
        Ok(ExternalMemory {
            external_memory,
            size,
            device: self.clone(),
            _file: ManuallyDrop::new(file),
        })
    }
}

/// An abstraction for imported external memory.
///
/// This struct can be created via [`CudaDevice::import_external_memory`].
/// The imported external memory will be destroyed when this struct is dropped.
#[derive(Debug)]
pub struct ExternalMemory {
    external_memory: sys::CUexternalMemory,
    size: u64,
    device: Arc<CudaDevice>,
    _file: ManuallyDrop<File>,
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        self.device.bind_to_thread().unwrap();

        unsafe { result::external_memory::destroy_external_memory(self.external_memory) }.unwrap();

        // From [CUDA docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735),
        // when successfully importing UNIX file descriptor:
        //
        // > Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully.
        // > Performing any operations on the file descriptor after it is imported results in undefined behavior.
        //
        // On the other hand, on Windows:
        //
        // > Ownership of this handle is not transferred to CUDA after the import operation,
        // > so the application must release the handle using the appropriate system call.
        //
        // Therefore, we manually drop the file when we are on Windows.
        #[cfg(windows)]
        unsafe {
            ManuallyDrop::<File>::drop(&mut self._file)
        };
    }
}

impl ExternalMemory {
    /// Map the whole external memory to get mapped buffer.
    pub fn map_all(self) -> Result<MappedBuffer, DriverError> {
        let size = self.size as usize;
        self.map_range(0..size)
    }

    /// Map a range of the external memory to a mapped buffer.
    ///
    /// Only one mapped buffer is allowed at a time.
    /// This is more restrictive than it necessarily needs to be,
    /// but it makes enforcing safety easier.
    ///
    /// # Panics
    /// This function will panic if the range is invalid,
    /// such as when the start or end is larger than the size.
    pub fn map_range(self, range: Range<usize>) -> Result<MappedBuffer, DriverError> {
        assert!(range.start as u64 <= self.size);
        assert!(range.end as u64 <= self.size);
        let device_ptr = unsafe {
            result::external_memory::get_mapped_buffer(
                self.external_memory,
                range.start as u64,
                range.len() as u64,
            )
        }?;
        Ok(MappedBuffer {
            device_ptr,
            len: range.len(),
            external_memory: self,
        })
    }
}

/// An abstraction for a mapped buffer for some external memory.
///
/// This struct can be created via [`ExternalMemory::map_range`] or [`ExternalMemory::map_all`].
/// The underlying mapped buffer will be freed when this struct is dropped.
#[derive(Debug)]
pub struct MappedBuffer {
    device_ptr: sys::CUdeviceptr,
    len: usize,
    external_memory: ExternalMemory,
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        self.external_memory.device.bind_to_thread().unwrap();
        unsafe { result::memory_free(self.device_ptr) }.unwrap()
    }
}

impl DeviceSlice<u8> for MappedBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

impl DevicePtr<u8> for MappedBuffer {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.device_ptr
    }
}
