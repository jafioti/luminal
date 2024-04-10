use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from_file("./examples/sin.ptx"), "sin", &["sin_kernel"])?;

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);

    let a_host = [1.0, 2.0, 3.0];
    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    // create a stream with `fork_default_stream()`
    // This synchronizes with the default stream, so since
    // we put this call **after** the `htod_copy` & `clone` above,
    // cuda will complete those orders **before** work on this stream
    // can start.
    let stream = dev.fork_default_stream()?;

    let f = dev.get_func("sin", "sin_kernel").unwrap();

    // we launch it differently too
    unsafe { f.launch_on_stream(&stream, cfg, (&mut b_dev, &a_dev, n as i32)) }?;

    // and we must join with the default work stream in order for copies
    // to work corrently.
    // NOTE: this is actually async with respect to the host!
    dev.wait_for(&stream)?;

    let a_host_2 = dev.sync_reclaim(a_dev)?;
    let b_host = dev.sync_reclaim(b_dev)?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
