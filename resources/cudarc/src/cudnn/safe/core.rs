use crate::{
    cudnn::{result, result::CudnnError, sys},
    driver::{CudaDevice, CudaStream},
};

use std::{marker::PhantomData, sync::Arc};

/// A handle to cuDNN.
///
/// This type is not send/sync because of <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#thread-safety>
#[derive(Debug)]
pub struct Cudnn {
    pub(crate) handle: sys::cudnnHandle_t,
    pub(crate) device: Arc<CudaDevice>,
}

impl Cudnn {
    /// Creates a new cudnn handle and sets the stream to the `device`'s stream.
    pub fn new(device: Arc<CudaDevice>) -> Result<Arc<Self>, CudnnError> {
        device.bind_to_thread().unwrap();
        let handle = result::create_handle()?;
        unsafe { result::set_stream(handle, device.stream as *mut _) }?;
        Ok(Arc::new(Self { handle, device }))
    }

    /// Sets the handle's current to either the stream specified, or the device's default work
    /// stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&self, opt_stream: Option<&CudaStream>) -> Result<(), CudnnError> {
        match opt_stream {
            Some(s) => result::set_stream(self.handle, s.stream as *mut _),
            None => result::set_stream(self.handle, self.device.stream as *mut _),
        }
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

/// Maps a rust type to a [sys::cudnnDataType_t]
pub trait CudnnDataType {
    const DATA_TYPE: sys::cudnnDataType_t;

    /// Certain CUDNN data types have a scaling parameter (usually called alpha/beta)
    /// that is a different type. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters)
    /// for more info, but basically f16 has a scalar of f32.
    type Scalar;

    /// Converts the type into the scaling parameter type. See [Self::Scalar].
    fn into_scaling_parameter(self) -> Self::Scalar;
}

macro_rules! cudnn_dtype {
    ($RustTy:ty, $CudnnTy:tt) => {
        impl CudnnDataType for $RustTy {
            const DATA_TYPE: sys::cudnnDataType_t = sys::cudnnDataType_t::$CudnnTy;
            type Scalar = Self;
            fn into_scaling_parameter(self) -> Self::Scalar {
                self
            }
        }
    };
}

cudnn_dtype!(f32, CUDNN_DATA_FLOAT);
cudnn_dtype!(f64, CUDNN_DATA_DOUBLE);
cudnn_dtype!(i8, CUDNN_DATA_INT8);
cudnn_dtype!(i32, CUDNN_DATA_INT32);
cudnn_dtype!(i64, CUDNN_DATA_INT64);
cudnn_dtype!(u8, CUDNN_DATA_UINT8);
cudnn_dtype!(bool, CUDNN_DATA_BOOLEAN);

#[cfg(feature = "f16")]
impl CudnnDataType for half::f16 {
    const DATA_TYPE: sys::cudnnDataType_t = sys::cudnnDataType_t::CUDNN_DATA_HALF;
    type Scalar = f32;
    fn into_scaling_parameter(self) -> Self::Scalar {
        self.to_f32()
    }
}
#[cfg(feature = "f16")]
impl CudnnDataType for half::bf16 {
    const DATA_TYPE: sys::cudnnDataType_t = sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16;
    type Scalar = f32;
    fn into_scaling_parameter(self) -> Self::Scalar {
        self.to_f32()
    }
}

/// A descriptor of a tensor. Create with:
/// 1. [`Cudnn::create_4d_tensor()`]
/// 2. [`Cudnn::create_4d_tensor_ex()`]
/// 3. [`Cudnn::create_nd_tensor()`]
#[derive(Debug)]
pub struct TensorDescriptor<T> {
    pub(crate) desc: sys::cudnnTensorDescriptor_t,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    /// Creates a 4d tensor descriptor.
    pub fn create_4d_tensor<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        format: sys::cudnnTensorFormat_t,
        dims: [std::ffi::c_int; 4],
    ) -> Result<TensorDescriptor<T>, CudnnError> {
        let desc = result::create_tensor_descriptor()?;
        let desc = TensorDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe { result::set_tensor4d_descriptor(desc.desc, format, T::DATA_TYPE, dims) }?;
        Ok(desc)
    }

    /// Creates a 4d tensor descriptor.
    pub fn create_4d_tensor_ex<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        dims: [std::ffi::c_int; 4],
        strides: [std::ffi::c_int; 4],
    ) -> Result<TensorDescriptor<T>, CudnnError> {
        let desc = result::create_tensor_descriptor()?;
        let desc = TensorDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe { result::set_tensor4d_descriptor_ex(desc.desc, T::DATA_TYPE, dims, strides) }?;
        Ok(desc)
    }

    /// Creates an nd (at LEAST 4d) tensor descriptor.
    pub fn create_nd_tensor<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        dims: &[std::ffi::c_int],
        strides: &[std::ffi::c_int],
    ) -> Result<TensorDescriptor<T>, CudnnError> {
        assert!(dims.len() >= 4);
        assert_eq!(dims.len(), strides.len());
        let desc = result::create_tensor_descriptor()?;
        let desc = TensorDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe {
            result::set_tensornd_descriptor(
                desc.desc,
                T::DATA_TYPE,
                dims.len() as std::ffi::c_int,
                dims.as_ptr(),
                strides.as_ptr(),
            )
        }?;
        Ok(desc)
    }
}

impl<T> Drop for TensorDescriptor<T> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_tensor_descriptor(desc) }.unwrap()
        }
    }
}
