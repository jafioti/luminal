use super::sys;
use core::ffi::{c_int, c_longlong, c_void};
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

/// Creates a handle to the cuBLAS library. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublascreate)
pub fn create_handle() -> Result<sys::cublasHandle_t, CublasError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasCreate_v2(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasdestroy)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cublasHandle_t) -> Result<(), CublasError> {
    sys::cublasDestroy_v2(handle).result()
}

/// Sets the stream cuBLAS will use. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream)
///
/// # Safety
///
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cublasHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    sys::cublasSetStream_v2(handle, stream).result()
}

/// Single precision matrix vector multiplication. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv)
///
/// # Safety
///
/// - `a`, `x`, and `y` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

/// Double precision matrix vector multiplication. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv)
///
/// # Safety
///
/// - `a`, `x`, and `y` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

#[cfg(feature = "f16")]
/// Half precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const half::f16,
    a: *const half::f16,
    lda: c_int,
    b: *const half::f16,
    ldb: c_int,
    beta: *const half::f16,
    c: *mut half::f16,
    ldc: c_int,
) -> Result<(), CublasError> {
    super::half::cublasHgemm(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

/// Single precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

/// Double precision matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

#[cfg(feature = "f16")]
/// Half precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const half::f16,
    a: *const half::f16,
    lda: c_int,
    stride_a: c_longlong,
    b: *const half::f16,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const half::f16,
    c: *mut half::f16,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    super::half::cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Single precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    stride_a: c_longlong,
    b: *const f32,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Double precision batched matmul. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemm_strided_batched(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    stride_a: c_longlong,
    b: *const f64,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
    stride_c: c_longlong,
    batch_size: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size,
    )
    .result()
}

/// Matmul with data types specified as parameters. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_ex(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: sys::cudaDataType,
    lda: c_int,
    b: *const c_void,
    b_type: sys::cudaDataType,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    c_type: sys::cudaDataType,
    ldc: c_int,
    compute_type: sys::cublasComputeType_t,
    algo: sys::cublasGemmAlgo_t,
) -> Result<(), CublasError> {
    sys::cublasGemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        compute_type,
        algo,
    )
    .result()
}

/// Strided batched matmul with data types specified as parameters. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex)
///
/// # Safety
///
/// - `a`, `b`, and `c` must be valid device pointers that have not been freed.
/// - `alpha` and `beta` can be pointers to host memory, but must be not null
/// - the strides and sizes must be sized correctly
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided_batched_ex(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: sys::cudaDataType,
    lda: c_int,
    stride_a: c_longlong,
    b: *const c_void,
    b_type: sys::cudaDataType,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const c_void,
    c: *mut c_void,
    c_type: sys::cudaDataType,
    ldc: c_int,
    stride_c: c_longlong,
    batch_count: c_int,
    compute_type: sys::cublasComputeType_t,
    algo: sys::cublasGemmAlgo_t,
) -> Result<(), CublasError> {
    sys::cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        stride_a,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )
    .result()
}
