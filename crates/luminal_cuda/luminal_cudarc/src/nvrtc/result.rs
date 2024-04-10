//! A thin wrapper around [sys] providing [Result]s with [NvrtcError].

use super::sys;
use core::{
    ffi::{c_char, c_int, CStr},
    mem::MaybeUninit,
};
use std::{ffi::CString, vec::Vec};

/// Wrapper around [sys::nvrtcResult]. See
/// [nvrtcResult docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1g31e41ef222c0ea75b4c48f715b3cd9f0)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvrtcError(pub sys::nvrtcResult);

impl sys::nvrtcResult {
    /// Transforms into a [Result] of [NvrtcError]
    pub fn result(self) -> Result<(), NvrtcError> {
        match self {
            sys::nvrtcResult::NVRTC_SUCCESS => Ok(()),
            _ => Err(NvrtcError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NvrtcError {}

/// Creates a program from source code `src`. This should be source code from a .cu file.
///
/// See [nvrtcCreateProgram() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1g9ae65f68911d1cf0adda2af4ad8cb458)
///
/// Example:
/// ```rust
/// # use cudarc::nvrtc::result::*;
/// let prog = create_program("extern \"C\" __global__ void kernel() { }").unwrap();
/// ```
pub fn create_program<S: AsRef<str>>(src: S) -> Result<sys::nvrtcProgram, NvrtcError> {
    let src_c = CString::new(src.as_ref()).unwrap();
    let mut prog = MaybeUninit::uninit();
    unsafe {
        sys::nvrtcCreateProgram(
            prog.as_mut_ptr(),
            src_c.as_c_str().as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
        .result()?;
        Ok(prog.assume_init())
    }
}

/// Compiles an already created program. Options should be of the form specified
/// in [nvrtc's supported compiler options](https://docs.nvidia.com/cuda/nvrtc/index.html#group__options).
///
/// See [nvrtcCompileProgram() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1g1f3136029db1413e362154b567297e8b)
///
/// Example:
///
/// ```rust
/// # use cudarc::nvrtc::result::*;
/// let prog = create_program("extern \"C\" __global__ void kernel() { }").unwrap();
/// unsafe { compile_program(prog, &["--ftz=true", "--fmad=true"]) }.unwrap();
/// ```
///
/// # Safety
///
/// `prog` must be created from [create_program()] and not have been freed by [destroy_program()].
pub unsafe fn compile_program<O: Clone + Into<Vec<u8>>>(
    prog: sys::nvrtcProgram,
    options: &[O],
) -> Result<(), NvrtcError> {
    let c_strings: Vec<CString> = options
        .iter()
        .cloned()
        .map(|o| CString::new(o).unwrap())
        .collect();
    let c_strs: Vec<&CStr> = c_strings.iter().map(CString::as_c_str).collect();
    let opts: Vec<*const c_char> = c_strs.iter().cloned().map(CStr::as_ptr).collect();
    sys::nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr()).result()
}

/// Releases resources associated with `prog`.
///
/// See [nvrtcDestroyProgram() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1gaa237c59615b7d4f48d5b308b5c9b140).
///
/// # Safety
///
/// `prog` must be created from [create_program()] and not have been freed by [destroy_program()].
pub unsafe fn destroy_program(prog: sys::nvrtcProgram) -> Result<(), NvrtcError> {
    sys::nvrtcDestroyProgram(&prog as *const _ as *mut _).result()
}

/// Extract the ptx associated with `prog`. Call [compile_program()] before this.
///
/// See [nvrtcGetPTX() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1gc9a66bbbd47c256f4a8955517b3965da)
/// and [nvrtcGetPTXSize() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1gc622d6ffb6fff71e209407da19612c1a).
///
/// # Safety
///
/// `prog` must be created from [create_program()] and not have been freed by [destroy_program()].
#[allow(clippy::slow_vector_initialization)]
pub unsafe fn get_ptx(prog: sys::nvrtcProgram) -> Result<Vec<c_char>, NvrtcError> {
    let mut size: usize = 0;
    sys::nvrtcGetPTXSize(prog, &mut size as *mut _).result()?;

    let mut ptx_src: Vec<c_char> = Vec::with_capacity(size);
    ptx_src.resize(size, 0);
    sys::nvrtcGetPTX(prog, ptx_src.as_mut_ptr()).result()?;
    Ok(ptx_src)
}

/// Extract log from a compiled program.
///
/// See [nvrtcGetProgramLog() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1g74c550e5cab81efbd59e4f72579edbd1)
/// and [nvrtcGetProgramLogSize() docs](https://docs.nvidia.com/cuda/nvrtc/index.html#group__compilation_1g59944bb118095ab53eec8994d056a18d).
///
/// # Safety
///
/// `prog` must be created from [create_program()] and not have been freed by [destroy_program()].
#[allow(clippy::slow_vector_initialization)]
pub unsafe fn get_program_log(prog: sys::nvrtcProgram) -> Result<Vec<c_char>, NvrtcError> {
    let mut size: usize = 0;
    sys::nvrtcGetProgramLogSize(prog, &mut size as *mut _).result()?;

    let mut log_src: Vec<c_char> = Vec::with_capacity(size);
    log_src.resize(size, 0);
    sys::nvrtcGetProgramLog(prog, log_src.as_mut_ptr()).result()?;
    Ok(log_src)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_program_no_opts() {
        let prog = create_program("extern \"C\" __global__ void kernel() { }").unwrap();
        unsafe { compile_program::<&str>(prog, &[]) }.unwrap();
        unsafe { destroy_program(prog) }.unwrap();
    }

    #[test]
    fn test_compile_program_1_opt() {
        let prog = create_program("extern \"C\" __global__ void kernel() { }").unwrap();
        unsafe { compile_program(prog, &["--ftz=true"]) }.unwrap();
        unsafe { destroy_program(prog) }.unwrap();
    }

    #[test]
    fn test_compile_program_2_opt() {
        let prog = create_program("extern \"C\" __global__ void kernel() { }").unwrap();
        unsafe { compile_program(prog, &["--ftz=true", "--fmad=true"]) }.unwrap();
        unsafe { destroy_program(prog) }.unwrap();
    }

    #[test]
    fn test_compile_bad_program() {
        let prog = create_program("extern \"C\" __global__ void kernel(").unwrap();
        assert_eq!(
            unsafe { compile_program::<&str>(prog, &[]) }.unwrap_err(),
            NvrtcError(sys::nvrtcResult::NVRTC_ERROR_COMPILATION)
        );
    }

    #[test]
    fn test_get_ptx() {
        const SRC: &str =
            "extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < numel) {
                out[i] = sin(inp[i]);
            }
        }";
        let prog = create_program(SRC).unwrap();
        unsafe { compile_program::<&str>(prog, &[]) }.unwrap();
        let ptx = unsafe { get_ptx(prog) }.unwrap();
        assert!(!ptx.is_empty());

        let log = unsafe { get_program_log(prog) }.unwrap();
        assert!(!log.is_empty());

        unsafe { destroy_program(prog) }.unwrap();
    }
}
