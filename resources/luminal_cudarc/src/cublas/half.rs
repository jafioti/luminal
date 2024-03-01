use super::sys;

use core::ffi::{c_int, c_longlong};

use half::f16;

extern "C" {
    pub fn cublasHgemm(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f16,
        A: *const f16,
        lda: c_int,
        B: *const f16,
        ldb: c_int,
        beta: *const f16,
        C: *mut f16,
        ldc: c_int,
    ) -> sys::cublasStatus_t;
}

extern "C" {
    pub fn cublasHgemmStridedBatched(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f16,
        A: *const f16,
        lda: c_int,
        strideA: c_longlong,
        B: *const f16,
        ldb: c_int,
        strideB: c_longlong,
        beta: *const f16,
        C: *mut f16,
        ldc: c_int,
        strideC: c_longlong,
        batchCount: c_int,
    ) -> sys::cublasStatus_t;
}
