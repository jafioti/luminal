use cudarc::driver::{CudaDevice, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // unsafe initialization of unset memory
    let _: CudaSlice<f32> = unsafe { dev.alloc::<f32>(10) }?;

    // this will have memory initialized as 0
    let _: CudaSlice<f64> = dev.alloc_zeros::<f64>(10)?;

    // initialize with a rust vec
    let _: CudaSlice<usize> = dev.htod_copy(vec![0; 10])?;

    // or finially, initialize with a slice. this is synchronous though.
    let _: CudaSlice<u32> = dev.htod_sync_copy(&[1, 2, 3])?;

    Ok(())
}
