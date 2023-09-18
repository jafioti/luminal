use cudarc::driver::{CudaDevice, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    let a: CudaSlice<f64> = dev.alloc_zeros::<f64>(10)?;
    let mut b = dev.alloc_zeros::<f64>(10)?;

    // you can do device to device copies of course
    dev.dtod_copy(&a, &mut b)?;

    // but also host to device copys with already allocated buffers
    dev.htod_copy_into(vec![2.0; 10], &mut b)?;

    // if you want to use slices, you can do synchronous copy
    dev.htod_sync_copy_into(&[3.0; 10], &mut b)?;

    // you can transfer back using reclaim:
    let mut a_host: Vec<f64> = dev.sync_reclaim(a)?;
    assert_eq!(a_host, [0.0; 10]);

    // or copy back without losing ownership:
    let b_host = dev.dtoh_sync_copy(&b)?;
    assert_eq!(b_host, [3.0; 10]);

    // or use a slice
    dev.dtoh_sync_copy_into(&b, &mut a_host)?;
    assert_eq!(a_host, b_host);

    Ok(())
}
