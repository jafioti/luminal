//! A thin wrapper around [sys] providing [Result]s with [CudnnError].

use std::mem::MaybeUninit;

use super::sys;

pub type CudnnResult<T> = Result<T, CudnnError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CudnnError(pub sys::cudnnStatus_t);

impl sys::cudnnStatus_t {
    /// Transforms into a [Result] of [CudnnError]
    pub fn result(self) -> Result<(), CudnnError> {
        match self {
            sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudnnError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudnnError {}

/// This function returns the version number of the cuDNN library. It returns the CUDNN_VERSION defined present in the cudnn.h header file.
///
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetVersion)
pub fn get_version() -> usize {
    unsafe { sys::cudnnGetVersion() }
}

/// The same version of a given cuDNN library can be compiled against different CUDA toolkit versions.
/// This routine returns the CUDA toolkit version that the currently used cuDNN library has been compiled against.
///
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetCudartVersion)
pub fn get_cudart_version() -> usize {
    unsafe { sys::cudnnGetCudartVersion() }
}

/// Runs all *VersionCheck functions.
pub fn version_check() -> Result<(), CudnnError> {
    unsafe {
        sys::cudnnAdvInferVersionCheck().result()?;
        sys::cudnnAdvTrainVersionCheck().result()?;
        sys::cudnnCnnInferVersionCheck().result()?;
        sys::cudnnCnnTrainVersionCheck().result()?;
        sys::cudnnOpsInferVersionCheck().result()?;
        sys::cudnnOpsTrainVersionCheck().result()?;
    }
    Ok(())
}

/// Creates a handle to the cuDNN library. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate)
pub fn create_handle() -> Result<sys::cudnnHandle_t, CudnnError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroy)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cudnnHandle_t) -> Result<(), CudnnError> {
    sys::cudnnDestroy(handle).result()
}

/// Sets the stream cuDNN will use. See
/// [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetStream)
///
/// # Safety
///
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cudnnHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetStream(handle, stream).result()
}

/// Allocates a new tensor descriptor.
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateTensorDescriptor)
pub fn create_tensor_descriptor() -> Result<sys::cudnnTensorDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateTensorDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// Sets data on a tensor descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptor)
///
/// # Safety
/// `tensor_desc` must have been created with [create_tensor_descriptor], and
/// NOT freed by [destroy_tensor_descriptor]
pub unsafe fn set_tensor4d_descriptor(
    tensor_desc: sys::cudnnTensorDescriptor_t,
    format: sys::cudnnTensorFormat_t,
    data_type: sys::cudnnDataType_t,
    [n, c, h, w]: [std::ffi::c_int; 4],
) -> Result<(), CudnnError> {
    sys::cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, n, c, h, w).result()
}

/// Sets data on a tensor descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptorEx)
///
/// # Safety
/// `tensor_desc` must have been created with [create_tensor_descriptor], and
/// NOT freed by [destroy_tensor_descriptor]
pub unsafe fn set_tensor4d_descriptor_ex(
    tensor_desc: sys::cudnnTensorDescriptor_t,
    data_type: sys::cudnnDataType_t,
    [n, c, h, w]: [std::ffi::c_int; 4],
    [n_stride, c_stride, h_stride, w_stride]: [std::ffi::c_int; 4],
) -> Result<(), CudnnError> {
    sys::cudnnSetTensor4dDescriptorEx(
        tensor_desc,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    )
    .result()
}

/// Sets data on a tensor descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor)
///
/// # Safety
/// `tensor_desc` must have been created with [create_tensor_descriptor], and
/// NOT freed by [destroy_tensor_descriptor]
pub unsafe fn set_tensornd_descriptor(
    tensor_desc: sys::cudnnTensorDescriptor_t,
    data_type: sys::cudnnDataType_t,
    num_dims: ::std::os::raw::c_int,
    dims: *const ::std::os::raw::c_int,
    strides: *const ::std::os::raw::c_int,
) -> Result<(), CudnnError> {
    sys::cudnnSetTensorNdDescriptor(tensor_desc, data_type, num_dims, dims, strides).result()
}

/// Destroys a tensor descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyTensorDescriptor)
///
/// # Safety
/// `desc` must NOT have been freed already.
pub unsafe fn destroy_tensor_descriptor(
    desc: sys::cudnnTensorDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyTensorDescriptor(desc).result()
}

/// Creates a filter descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateFilterDescriptor)
pub fn create_filter_descriptor() -> Result<sys::cudnnFilterDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateFilterDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// Sets data on a pre allocated filter descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilter4dDescriptor)
///
/// # Safety
/// `filter_desc` must be have been allocated with [create_filter_descriptor]
/// and NOT already freed by [destroy_filter_descriptor].
pub unsafe fn set_filter4d_descriptor(
    filter_desc: sys::cudnnFilterDescriptor_t,
    data_type: sys::cudnnDataType_t,
    format: sys::cudnnTensorFormat_t,
    [k, c, h, w]: [::std::os::raw::c_int; 4],
) -> Result<(), CudnnError> {
    sys::cudnnSetFilter4dDescriptor(filter_desc, data_type, format, k, c, h, w).result()
}

/// Destroys a filter descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyFilterDescriptor)
///
/// # Safety
/// `desc` must NOT have already been freed.
pub unsafe fn destroy_filter_descriptor(
    desc: sys::cudnnFilterDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyFilterDescriptor(desc).result()
}

/// Allocates a convolution descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateConvolutionDescriptor).
pub fn create_convolution_descriptor() -> Result<sys::cudnnConvolutionDescriptor_t, CudnnError> {
    let mut desc = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateConvolutionDescriptor(desc.as_mut_ptr()).result()?;
        Ok(desc.assume_init())
    }
}

/// Sets data on a conv descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolution2dDescriptor)
///
/// # Safety
/// `conv_desc` must have been allocated by [create_convolution_descriptor]
/// and NOT freed by [destroy_convolution_descriptor].
#[allow(clippy::too_many_arguments)]
pub unsafe fn set_convolution2d_descriptor(
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    pad_h: std::ffi::c_int,
    pad_w: std::ffi::c_int,
    u: std::ffi::c_int,
    v: std::ffi::c_int,
    dilation_h: std::ffi::c_int,
    dilation_w: std::ffi::c_int,
    mode: sys::cudnnConvolutionMode_t,
    compute_type: sys::cudnnDataType_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h,
        pad_w,
        u,
        v,
        dilation_h,
        dilation_w,
        mode,
        compute_type,
    )
    .result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionMathType).
/// # Safety
/// `desc` must NOT have been freed already
pub unsafe fn set_convolution_math_type(
    desc: sys::cudnnConvolutionDescriptor_t,
    math_type: sys::cudnnMathType_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetConvolutionMathType(desc, math_type).result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolutionGroupCount)
/// # Safety
/// `desc` must NOT have been freed already
pub unsafe fn set_convolution_group_count(
    desc: sys::cudnnConvolutionDescriptor_t,
    group_count: i32,
) -> Result<(), CudnnError> {
    sys::cudnnSetConvolutionGroupCount(desc, group_count).result()
}

/// Destroys a descriptor. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyConvolutionDescriptor).
/// # Safety
/// `desc` must NOT have been already freed.
pub unsafe fn destroy_convolution_descriptor(
    desc: sys::cudnnConvolutionDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyConvolutionDescriptor(desc).result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardAlgorithm_v7)
///
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_forward_algorithm(
    handle: sys::cudnnHandle_t,
    src: sys::cudnnTensorDescriptor_t,
    filter: sys::cudnnFilterDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    dest: sys::cudnnTensorDescriptor_t,
    requested_algo_count: std::ffi::c_int,
    returned_algo_count: *mut std::ffi::c_int,
    perf_results: *mut sys::cudnnConvolutionFwdAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        src,
        filter,
        conv,
        dest,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardWorkspaceSize)
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
pub unsafe fn get_convolution_forward_workspace_size(
    handle: sys::cudnnHandle_t,
    x: sys::cudnnTensorDescriptor_t,
    w: sys::cudnnFilterDescriptor_t,
    conv: sys::cudnnConvolutionDescriptor_t,
    y: sys::cudnnTensorDescriptor_t,
    algo: sys::cudnnConvolutionFwdAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x,
        w,
        conv,
        y,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// Launch the conv forward kernel.
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward).
///
/// # Safety
/// - handles and descriptors must still be allocated
/// - all pointers must be valid data pointers
/// - the format of descriptors should match the data allocated
///   in the pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_forward(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    x_desc: sys::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    w_desc: sys::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    y_desc: sys::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionForward(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        y_desc,
        y,
    )
    .result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardDataAlgorithm_v7)
///
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_backward_data_algorithm(
    handle: sys::cudnnHandle_t,
    w_desc: sys::cudnnFilterDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    dx_desc: sys::cudnnTensorDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: *mut ::std::os::raw::c_int,
    perf_results: *mut sys::cudnnConvolutionBwdDataAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**. See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardDataWorkspaceSize)
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
pub unsafe fn get_convolution_backward_data_workspace_size(
    handle: sys::cudnnHandle_t,
    w_desc: sys::cudnnFilterDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    dx_desc: sys::cudnnTensorDescriptor_t,
    algo: sys::cudnnConvolutionBwdDataAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// Launch the backward data kernel.
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData).
///
/// # Safety
/// - handles and descriptors must still be allocated
/// - all pointers must be valid data pointers
/// - the format of descriptors should match the data allocated
///   in the pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_backward_data(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    w_desc: sys::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dy_desc: sys::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionBwdDataAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    dx_desc: sys::cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionBackwardData(
        handle,
        alpha,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dx_desc,
        dx,
    )
    .result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardFilterAlgorithm_v7)
///
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_backward_filter_algorithm(
    handle: sys::cudnnHandle_t,
    src_desc: sys::cudnnTensorDescriptor_t,
    diff_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    grad_desc: sys::cudnnFilterDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: *mut ::std::os::raw::c_int,
    perf_results: *mut sys::cudnnConvolutionBwdFilterAlgoPerf_t,
) -> Result<(), CudnnError> {
    sys::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle,
        src_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )
    .result()
}

/// Returns size in **bytes**.
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionBackwardFilterWorkspaceSize)
/// # Safety
/// - All handles & descriptors must still be allocated.
/// - The pointers must point to valid memory.
pub unsafe fn get_convolution_backward_filter_workspace_size(
    handle: sys::cudnnHandle_t,
    x_desc: sys::cudnnTensorDescriptor_t,
    dy_desc: sys::cudnnTensorDescriptor_t,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    grad_desc: sys::cudnnFilterDescriptor_t,
    algo: sys::cudnnConvolutionBwdFilterAlgo_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        x_desc,
        dy_desc,
        conv_desc,
        grad_desc,
        algo,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// Launch the backward data kernel.
/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter).
///
/// # Safety
/// - handles and descriptors must still be allocated
/// - all pointers must be valid data pointers
/// - the format of descriptors should match the data allocated
///   in the pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolution_backward_filter(
    handle: sys::cudnnHandle_t,
    alpha: *const ::core::ffi::c_void,
    x_desc: sys::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dy_desc: sys::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: sys::cudnnConvolutionDescriptor_t,
    algo: sys::cudnnConvolutionBwdFilterAlgo_t,
    work_space: *mut ::core::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const ::core::ffi::c_void,
    dw_desc: sys::cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnConvolutionBackwardFilter(
        handle,
        alpha,
        x_desc,
        x,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dw_desc,
        dw,
    )
    .result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateReduceTensorDescriptor).
pub fn create_reduce_tensor_descriptor() -> Result<sys::cudnnReduceTensorDescriptor_t, CudnnError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cudnnCreateReduceTensorDescriptor(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetReduceTensorDescriptor)
///
/// # Safety
/// All the descriptors must be allocated properly and not have been destroyed.
pub unsafe fn set_reduce_tensor_descriptor(
    tensor_desc: sys::cudnnReduceTensorDescriptor_t,
    tensor_op: sys::cudnnReduceTensorOp_t,
    tensor_comp_type: sys::cudnnDataType_t,
    tensor_nan_opt: sys::cudnnNanPropagation_t,
    tensor_indices: sys::cudnnReduceTensorIndices_t,
    tensor_indices_type: sys::cudnnIndicesType_t,
) -> Result<(), CudnnError> {
    sys::cudnnSetReduceTensorDescriptor(
        tensor_desc,
        tensor_op,
        tensor_comp_type,
        tensor_nan_opt,
        tensor_indices,
        tensor_indices_type,
    )
    .result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyReduceTensorDescriptor).
///
/// # Safety
/// Descriptor must not have been freed already.
pub unsafe fn destroy_reduce_tensor_descriptor(
    tensor_desc: sys::cudnnReduceTensorDescriptor_t,
) -> Result<(), CudnnError> {
    sys::cudnnDestroyReduceTensorDescriptor(tensor_desc).result()
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetReductionIndicesSize)
///
/// # Safety
/// Handle and descriptor must be valid (properly allocated and not freed already).
pub unsafe fn get_reduction_indices_size(
    handle: sys::cudnnHandle_t,
    reduce_tensor_desc: sys::cudnnReduceTensorDescriptor_t,
    a_desc: sys::cudnnTensorDescriptor_t,
    c_desc: sys::cudnnTensorDescriptor_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetReductionIndicesSize(
        handle,
        reduce_tensor_desc,
        a_desc,
        c_desc,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetReductionWorkspaceSize)
///
/// # Safety
/// Handle and descriptors must be properly allocated and not freed already.
pub unsafe fn get_reduction_workspace_size(
    handle: sys::cudnnHandle_t,
    reduce_tensor_desc: sys::cudnnReduceTensorDescriptor_t,
    a_desc: sys::cudnnTensorDescriptor_t,
    c_desc: sys::cudnnTensorDescriptor_t,
) -> Result<usize, CudnnError> {
    let mut size_in_bytes = [0];
    sys::cudnnGetReductionWorkspaceSize(
        handle,
        reduce_tensor_desc,
        a_desc,
        c_desc,
        size_in_bytes.as_mut_ptr(),
    )
    .result()?;
    Ok(size_in_bytes[0])
}

/// See [nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnReduceTensor)
///
/// # Safety
/// - All data must be properly allocated and not freed.
/// - The descriptors must be the same data type as the pointers
/// - Misuse of this function could result in out of bounds memory accesses.
#[allow(clippy::too_many_arguments)]
pub unsafe fn reduce_tensor(
    handle: sys::cudnnHandle_t,
    reduce_tensor_desc: sys::cudnnReduceTensorDescriptor_t,
    indices: *mut std::ffi::c_void,
    indices_size_in_bytes: usize,
    workspace: *mut std::ffi::c_void,
    workspace_size_in_bytes: usize,
    alpha: *const std::ffi::c_void,
    a_desc: sys::cudnnTensorDescriptor_t,
    a: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c_desc: sys::cudnnTensorDescriptor_t,
    c: *mut std::ffi::c_void,
) -> Result<(), CudnnError> {
    sys::cudnnReduceTensor(
        handle,
        reduce_tensor_desc,
        indices,
        indices_size_in_bytes,
        workspace,
        workspace_size_in_bytes,
        alpha,
        a_desc,
        a,
        beta,
        c_desc,
        c,
    )
    .result()
}
