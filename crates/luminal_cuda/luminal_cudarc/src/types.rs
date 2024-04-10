//! Exposes [CudaTypeName] which maps between rust type names
//! and the corresponding cuda kernel type names.
//!
//! For example, `f32` in rust corresponds to `float` in a cuda
//! kernel.

/// Maps a rust type to it's corresponding [CudaTypeName::NAME] in cuda c++ land.
pub trait CudaTypeName {
    const NAME: &'static str;
}

macro_rules! cuda_type {
    ($RustTy:ty, $CudaTy:expr) => {
        impl CudaTypeName for $RustTy {
            const NAME: &'static str = $CudaTy;
        }
    };
}

cuda_type!(bool, "bool");
cuda_type!(i8, "char");
cuda_type!(i16, "short");
cuda_type!(i32, "int");
cuda_type!(i64, "long");
cuda_type!(isize, "intptr_t");
cuda_type!(u8, "unsigned char");
cuda_type!(u16, "unsigned short");
cuda_type!(u32, "unsigned int");
cuda_type!(u64, "unsigned long");
cuda_type!(usize, "size_t");
cuda_type!(f32, "float");
cuda_type!(f64, "double");
#[cfg(feature = "f16")]
cuda_type!(half::f16, "__half");
#[cfg(feature = "f16")]
cuda_type!(half::bf16, "__nv_bfloat16");
