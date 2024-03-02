use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}
";

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "matmul", &["matmul"])?;
    let f = dev.get_func("matmul", "matmul").unwrap();
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];

    let a_dev = dev.htod_sync_copy(&a_host)?;
    let b_dev = dev.htod_sync_copy(&b_host)?;
    let mut c_dev = dev.htod_sync_copy(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let cfg = LaunchConfig {
        block_dim: (2, 2, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, 2i32)) }?;

    dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}
