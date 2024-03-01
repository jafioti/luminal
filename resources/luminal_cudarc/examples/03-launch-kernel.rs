use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // You can load a function from a pre-compiled PTX like so:
    dev.load_ptx(Ptx::from_file("./examples/sin.ptx"), "sin", &["sin_kernel"])?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("sin", "sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    let n = 3;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }?;

    let a_host_2 = dev.sync_reclaim(a_dev)?;
    let b_host = dev.sync_reclaim(b_dev)?;

    println!("Found {:?}", b_host);
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
