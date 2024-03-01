use super::core::*;
use crate::{
    cudnn::{result, result::CudnnError, sys},
    driver::{DevicePtr, DevicePtrMut},
};

use std::{marker::PhantomData, sync::Arc};

/// A marker type used with [ReductionDescriptor] to indicate the
/// reduction operation should return flattened indices. Corresponds
/// to [sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_FLATTENED_INDICES].
#[derive(Debug, Default, Copy, Clone)]
pub struct FlatIndices;

/// A marker type used with [ReductionDescriptor] to indicate the
/// reduction operation should **NOT** return indices. Corresponds
/// to [sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES].
#[derive(Debug, Default, Copy, Clone)]
pub struct NoIndices;

/// A reduction descriptor. Create with [`Cudnn::create_reduction_with_indices()`] if you
/// want the indices returned, or [`Cudnn::create_reduction_without_indices()`] if not.
#[derive(Debug)]
pub struct ReductionDescriptor<T, Idx> {
    pub(crate) desc: sys::cudnnReduceTensorDescriptor_t,
    #[allow(unused)]
    pub(crate) indices: Idx,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    /// Create a reduction descriptor that computes indices.
    pub fn create_reduction_flat_indices<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        op: sys::cudnnReduceTensorOp_t,
        nan_opt: sys::cudnnNanPropagation_t,
    ) -> Result<ReductionDescriptor<T, FlatIndices>, CudnnError> {
        let desc = result::create_reduce_tensor_descriptor()?;
        let desc = ReductionDescriptor {
            desc,
            indices: FlatIndices,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe {
            result::set_reduce_tensor_descriptor(
                desc.desc,
                op,
                T::DATA_TYPE,
                nan_opt,
                sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                sys::cudnnIndicesType_t::CUDNN_32BIT_INDICES,
            )
        }?;
        Ok(desc)
    }

    /// Create a reduction descriptor that does NOT compute indices.
    pub fn create_reduction_no_indices<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        op: sys::cudnnReduceTensorOp_t,
        nan_opt: sys::cudnnNanPropagation_t,
    ) -> Result<ReductionDescriptor<T, NoIndices>, CudnnError> {
        let desc = result::create_reduce_tensor_descriptor()?;
        let desc = ReductionDescriptor {
            desc,
            indices: NoIndices,
            handle: self.clone(),
            marker: PhantomData,
        };
        unsafe {
            result::set_reduce_tensor_descriptor(
                desc.desc,
                op,
                T::DATA_TYPE,
                nan_opt,
                sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES,
                sys::cudnnIndicesType_t::CUDNN_32BIT_INDICES,
            )
        }?;
        Ok(desc)
    }
}

impl<T, Idx> Drop for ReductionDescriptor<T, Idx> {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_reduce_tensor_descriptor(desc) }.unwrap()
        }
    }
}

/// A reduction operation. Pass in fields directly, and then call launch.
pub struct ReduceTensor<'a, T: CudnnDataType, Idx> {
    /// The reduction descriptor.
    pub reduce: &'a ReductionDescriptor<T, Idx>,
    /// The input tensor
    pub a: &'a TensorDescriptor<T>,
    /// The output tensor
    pub c: &'a TensorDescriptor<T>,
}

impl<'a, T: CudnnDataType> ReduceTensor<'a, T, FlatIndices> {
    /// Get's the size of the indices tensor required for this operation.
    ///
    /// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetReductionIndicesSize).
    pub fn get_indices_size(&self) -> Result<usize, CudnnError> {
        unsafe {
            result::get_reduction_indices_size(
                self.reduce.handle.handle,
                self.reduce.desc,
                self.a.desc,
                self.c.desc,
            )
        }
    }
}

impl<'a, T: CudnnDataType, Idx> ReduceTensor<'a, T, Idx> {
    /// Gets the size of the workspace for this operation.
    ///
    /// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetReductionWorkspaceSize)
    pub fn get_workspace_size(&self) -> Result<usize, CudnnError> {
        unsafe {
            result::get_reduction_workspace_size(
                self.reduce.handle.handle,
                self.reduce.desc,
                self.a.desc,
                self.c.desc,
            )
        }
    }
}

impl<'a, T: CudnnDataType> ReduceTensor<'a, T, FlatIndices> {
    /// Launches the operation with indices.
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self`.
    pub unsafe fn launch<Indices, Workspace, A, C>(
        &self,
        indices: &mut Indices,
        workspace: &mut Workspace,
        (alpha, beta): (T, T),
        a: &A,
        c: &mut C,
    ) -> Result<(), CudnnError>
    where
        Indices: DevicePtrMut<u32>,
        Workspace: DevicePtrMut<u8>,
        A: DevicePtr<T>,
        C: DevicePtrMut<T>,
    {
        result::reduce_tensor(
            self.reduce.handle.handle,
            self.reduce.desc,
            *indices.device_ptr_mut() as *mut std::ffi::c_void,
            indices.num_bytes(),
            *workspace.device_ptr_mut() as *mut std::ffi::c_void,
            workspace.num_bytes(),
            (&alpha) as *const T as *const std::ffi::c_void,
            self.a.desc,
            *a.device_ptr() as *const _,
            (&beta) as *const T as *const std::ffi::c_void,
            self.c.desc,
            *c.device_ptr_mut() as *mut _,
        )
    }
}

impl<'a, T: CudnnDataType> ReduceTensor<'a, T, NoIndices> {
    /// Launches the operation with no indices.
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self`.
    pub unsafe fn launch<Workspace, A, C>(
        &self,
        workspace: &mut Workspace,
        (alpha, beta): (T, T),
        a: &A,
        c: &mut C,
    ) -> Result<(), CudnnError>
    where
        Workspace: DevicePtrMut<u8>,
        A: DevicePtr<T>,
        C: DevicePtrMut<T>,
    {
        result::reduce_tensor(
            self.reduce.handle.handle,
            self.reduce.desc,
            std::ptr::null_mut(),
            0,
            *workspace.device_ptr_mut() as *mut std::ffi::c_void,
            workspace.num_bytes(),
            (&alpha) as *const T as *const std::ffi::c_void,
            self.a.desc,
            *a.device_ptr() as *const _,
            (&beta) as *const T as *const std::ffi::c_void,
            self.c.desc,
            *c.device_ptr_mut() as *mut _,
        )
    }
}
