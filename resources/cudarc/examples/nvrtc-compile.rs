use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};

fn main() -> Result<(), CompileError> {
    let opts = CompileOptions {
        ftz: Some(true),
        prec_div: Some(false),
        prec_sqrt: Some(false),
        fmad: Some(true),
        ..Default::default()
    };

    let _ = compile_ptx_with_opts(
        "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
        opts,
    )?;
    println!("Compilation succeeded!");
    Ok(())
}
