use super::sys;
use crate::cublaslt::sys::cublasLtMatmulAlgo_t;
use core::ffi::c_void;
use core::mem::MaybeUninit;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub sys::cublasStatus_t);

impl sys::cublasStatus_t {
    pub fn result(self) -> Result<(), CublasError> {
        match self {
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

/// Creates a handle to the cuBLASLT library. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltcreate)
pub fn create_handle() -> Result<sys::cublasLtHandle_t, CublasError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasLtCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltdestroy)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cublasLtHandle_t) -> Result<(), CublasError> {
    sys::cublasLtDestroy(handle).result()
}

/// Creates a matrix layout descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutcreate)
pub fn create_matrix_layout(
    matrix_type: sys::cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> Result<sys::cublasLtMatrixLayout_t, CublasError> {
    let mut matrix_layout = MaybeUninit::uninit();
    unsafe {
        sys::cublasLtMatrixLayoutCreate(matrix_layout.as_mut_ptr(), matrix_type, rows, cols, ld)
            .result()?;
        Ok(matrix_layout.assume_init())
    }
}

/// Sets the value of the specified attribute belonging to a previously created matrix layout
/// descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutsetattribute)
///
/// # Safety
/// `matrix_layout` must not have been freed already.
pub unsafe fn set_matrix_layout_attribute(
    matrix_layout: sys::cublasLtMatrixLayout_t,
    attr: sys::cublasLtMatrixLayoutAttribute_t,
    buf: *const c_void,
    buf_size: usize,
) -> Result<(), CublasError> {
    sys::cublasLtMatrixLayoutSetAttribute(matrix_layout, attr, buf, buf_size).result()
}

/// Destroys a matrix layout previously created with [create_matrix_layout(...)]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutdestroy)
///
/// # Safety
///
/// `matrix_layout` must not have been freed already.
pub unsafe fn destroy_matrix_layout(
    matrix_layout: sys::cublasLtMatrixLayout_t,
) -> Result<(), CublasError> {
    sys::cublasLtMatrixLayoutDestroy(matrix_layout).result()
}

/// Creates a matrix multiply descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmuldesccreate)
pub fn create_matmul_desc(
    compute_type: sys::cublasComputeType_t,
    scale_type: sys::cudaDataType,
) -> Result<sys::cublasLtMatmulDesc_t, CublasError> {
    let mut matmul_desc = MaybeUninit::uninit();
    unsafe {
        sys::cublasLtMatmulDescCreate(matmul_desc.as_mut_ptr(), compute_type, scale_type)
            .result()?;
        Ok(matmul_desc.assume_init())
    }
}

/// Sets the value of the specified attribute belonging to a previously created matrix multiply
/// descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmuldescsetattribute)
///
/// # Safety
/// `matmul_desc` must not be freed already.
pub unsafe fn set_matmul_desc_attribute(
    matmul_desc: sys::cublasLtMatmulDesc_t,
    attr: sys::cublasLtMatmulDescAttributes_t,
    buf: *const c_void,
    buf_size: usize,
) -> Result<(), CublasError> {
    sys::cublasLtMatmulDescSetAttribute(matmul_desc, attr, buf, buf_size).result()
}

/// Destroys a matrix multiply descriptor previously created with [create_matmul_desc(...)]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmuldescdestroy)
///
/// # Safety
///
/// `matmul_desc` must not have been freed already.
pub unsafe fn destroy_matmul_desc(
    matmul_desc: sys::cublasLtMatmulDesc_t,
) -> Result<(), CublasError> {
    sys::cublasLtMatmulDescDestroy(matmul_desc).result()
}

/// Creates a matrix multiply heuristic search preferences descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulpreferencecreate)
pub fn create_matmul_pref() -> Result<sys::cublasLtMatmulPreference_t, CublasError> {
    let mut matmul_pref = MaybeUninit::uninit();
    unsafe {
        sys::cublasLtMatmulPreferenceCreate(matmul_pref.as_mut_ptr()).result()?;
        Ok(matmul_pref.assume_init())
    }
}

/// Sets the value of the specified attribute belonging to a previously create matrix multiply
/// preferences descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulpreferencesetattribute)
///
/// # Safety
/// `matmul_pref` must not have been freed already.
pub unsafe fn set_matmul_pref_attribute(
    matmul_pref: sys::cublasLtMatmulPreference_t,
    attr: sys::cublasLtMatmulPreferenceAttributes_t,
    buf: *const c_void,
    buf_size: usize,
) -> Result<(), CublasError> {
    sys::cublasLtMatmulPreferenceSetAttribute(matmul_pref, attr, buf, buf_size).result()
}

/// Destroys a matrix multiply preferences descriptor previously created
/// with [create_matmul_pref()]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulpreferencedestroy)
///
/// # Safety
///
/// `matmul_pref` must not have been freed already.
pub unsafe fn destroy_matmul_pref(
    matmul_pref: sys::cublasLtMatmulPreference_t,
) -> Result<(), CublasError> {
    sys::cublasLtMatmulPreferenceDestroy(matmul_pref).result()
}

/// Retrieves the fastest possible algorithm for the matrix multiply operation function
/// given input matrices A, B and C and the output matrix D. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic)
///
/// # Safety
/// All the parameters must not have been freed already & must be valid layouts for allocations.
pub unsafe fn get_matmul_algo_heuristic(
    handle: sys::cublasLtHandle_t,
    matmul_desc: sys::cublasLtMatmulDesc_t,
    a_layout: sys::cublasLtMatrixLayout_t,
    b_layout: sys::cublasLtMatrixLayout_t,
    c_layout: sys::cublasLtMatrixLayout_t,
    d_layout: sys::cublasLtMatrixLayout_t,
    matmul_pref: sys::cublasLtMatmulPreference_t,
) -> Result<sys::cublasLtMatmulHeuristicResult_t, CublasError> {
    let mut matmul_heuristic = MaybeUninit::uninit();
    let mut algo_count = 0;

    sys::cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        d_layout,
        matmul_pref,
        1, // only select the fastest algo
        matmul_heuristic.as_mut_ptr(),
        &mut algo_count,
    )
    .result()?;

    if algo_count == 0 {
        return Err(CublasError(
            sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
        ));
    }

    let matmul_heuristic = matmul_heuristic.assume_init();
    matmul_heuristic.state.result()?;

    Ok(matmul_heuristic)
}

/// Computes the matrix multiplication of matrics A and B to produce the output matrix D,
/// according to the following operation: D = alpha*(A*B) + beta*(C)
/// where A, B, and C are input matrices, and alpha and beta are input scalars. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
///
/// # Safety
/// All the sys objects can't have been freed already.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul(
    handle: sys::cublasLtHandle_t,
    matmul_desc: sys::cublasLtMatmulDesc_t,
    alpha: *const c_void,
    beta: *const c_void,
    a: *const c_void,
    a_layout: sys::cublasLtMatrixLayout_t,
    b: *const c_void,
    b_layout: sys::cublasLtMatrixLayout_t,
    c: *const c_void,
    c_layout: sys::cublasLtMatrixLayout_t,
    d: *mut c_void,
    d_layout: sys::cublasLtMatrixLayout_t,
    algo: *const cublasLtMatmulAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    sys::cublasLtMatmul(
        handle,
        matmul_desc,
        alpha,
        a,
        a_layout,
        b,
        b_layout,
        beta,
        c,
        c_layout,
        d,
        d_layout,
        algo,
        workspace,
        workspace_size,
        stream,
    )
    .result()
}
