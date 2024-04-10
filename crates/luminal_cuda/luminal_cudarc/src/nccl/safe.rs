use super::{result, sys};
use crate::driver::{CudaDevice, CudaSlice};
use std::mem::MaybeUninit;
use std::ptr;
use std::{sync::Arc, vec, vec::Vec};

pub use result::{group_end, group_start};

#[derive(Debug)]
pub struct Comm {
    comm: sys::ncclComm_t,
    device: Arc<CudaDevice>,
    rank: usize,
    world_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Id {
    id: sys::ncclUniqueId,
}

impl Id {
    pub fn new() -> Result<Self, result::NcclError> {
        let id = result::get_uniqueid()?;
        Ok(Self { id })
    }

    pub fn uninit(internal: [::core::ffi::c_char; 128usize]) -> Self {
        let id = sys::ncclUniqueId { internal };
        Self { id }
    }

    pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
        &self.id.internal
    }
}

pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

fn convert_to_nccl_reduce_op(op: &ReduceOp) -> sys::ncclRedOp_t {
    match op {
        ReduceOp::Sum => sys::ncclRedOp_t::ncclSum,
        ReduceOp::Prod => sys::ncclRedOp_t::ncclProd,
        ReduceOp::Max => sys::ncclRedOp_t::ncclMax,
        ReduceOp::Min => sys::ncclRedOp_t::ncclMin,
        ReduceOp::Avg => sys::ncclRedOp_t::ncclAvg,
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        // TODO(thenerdstation): Shoule we instead do finalize then destory?
        unsafe {
            result::comm_abort(self.comm).expect("Error when aborting Comm.");
        }
    }
}

pub trait NcclType {
    fn as_nccl_type() -> sys::ncclDataType_t;
}

macro_rules! define_nccl_type {
    ($t:ty, $nccl_type:expr) => {
        impl NcclType for $t {
            fn as_nccl_type() -> sys::ncclDataType_t {
                $nccl_type
            }
        }
    };
}

define_nccl_type!(f32, sys::ncclDataType_t::ncclFloat32);
define_nccl_type!(f64, sys::ncclDataType_t::ncclFloat64);
define_nccl_type!(i8, sys::ncclDataType_t::ncclInt8);
define_nccl_type!(i32, sys::ncclDataType_t::ncclInt32);
define_nccl_type!(i64, sys::ncclDataType_t::ncclInt64);
define_nccl_type!(u8, sys::ncclDataType_t::ncclUint8);
define_nccl_type!(u32, sys::ncclDataType_t::ncclUint32);
define_nccl_type!(u64, sys::ncclDataType_t::ncclUint64);
define_nccl_type!(char, sys::ncclDataType_t::ncclUint8);
#[cfg(feature = "f16")]
define_nccl_type!(half::f16, sys::ncclDataType_t::ncclFloat16);
#[cfg(feature = "f16")]
define_nccl_type!(half::bf16, sys::ncclDataType_t::ncclBfloat16);
impl Comm {
    /// Primitive to create new communication link on a single thread.
    /// WARNING: You are likely to get limited throughput using a single core
    /// to control multiple GPUs
    /// ```
    /// # use cudarc::driver::safe::{CudaDevice};
    /// # use cudarc::nccl::safe::{Comm, ReduceOp, group_start, group_end};
    /// let n = 2;
    /// let n_devices = CudaDevice::count().unwrap() as usize;
    /// let devices : Vec<_> = (0..n_devices).flat_map(CudaDevice::new).collect();
    /// let comms = Comm::from_devices(devices).unwrap();
    /// group_start().unwrap();
    /// (0..n_devices).map(|i| {
    ///     let comm = &comms[i];
    ///     let dev = comm.device();
    ///     let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
    ///     let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    ///     comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///         .unwrap();

    /// });
    /// group_start().unwrap();
    /// ```
    pub fn from_devices(devices: Vec<Arc<CudaDevice>>) -> Result<Vec<Self>, result::NcclError> {
        let n_devices = devices.len();
        let mut comms = vec![std::ptr::null_mut(); n_devices];
        let ordinals: Vec<_> = devices.iter().map(|d| d.ordinal as i32).collect();
        unsafe {
            result::comm_init_all(comms.as_mut_ptr(), n_devices as i32, ordinals.as_ptr())?;
        }

        let comms: Vec<Self> = comms
            .into_iter()
            .zip(devices.iter().cloned())
            .enumerate()
            .map(|(rank, (comm, device))| Self {
                comm,
                device,
                rank,
                world_size: n_devices,
            })
            .collect();

        Ok(comms)
    }

    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Primitive to create new communication link on each process (threads are possible but not
    /// recommended).
    ///
    /// WARNING: If using threads, uou are likely to get limited throughput using a single core
    /// to control multiple GPUs. Cuda drivers effectively use a global mutex thrashing
    /// performance on multi threaded multi GPU.
    /// ```
    /// # use cudarc::driver::safe::{CudaDevice};
    /// # use cudarc::nccl::safe::{Comm, Id, ReduceOp};
    /// let n = 2;
    /// let n_devices = 1; // This is to simplify this example.
    /// // Spawn this only on rank 0
    /// let id = Id::new().unwrap();
    /// // Send id.internal() to other ranks
    /// // let id = Id::uninit(id.internal().clone()); on other ranks
    ///
    /// let rank = 0;
    /// let dev = CudaDevice::new(rank).unwrap();
    /// let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();
    /// let slice = dev.htod_copy(vec![(rank + 1) as f32 * 1.0; n]).unwrap();
    /// let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    /// comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///     .unwrap();

    /// let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

    /// assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
    /// ```
    pub fn from_rank(
        device: Arc<CudaDevice>,
        rank: usize,
        world_size: usize,
        id: Id,
    ) -> Result<Self, result::NcclError> {
        let mut comm = MaybeUninit::uninit();

        let comm = unsafe {
            result::comm_init_rank(
                comm.as_mut_ptr(),
                world_size
                    .try_into()
                    .expect("World_size cannot be casted to i32"),
                id.id,
                rank.try_into().expect("Rank cannot be cast to i32"),
            )?;
            comm.assume_init()
        };
        Ok(Self {
            comm,
            device,
            rank,
            world_size,
        })
    }
}

impl Comm {
    pub fn send<T: NcclType>(
        &self,
        data: &CudaSlice<T>,
        peer: i32,
    ) -> Result<(), result::NcclError> {
        unsafe {
            result::send(
                data.cu_device_ptr as *mut _,
                data.len,
                T::as_nccl_type(),
                peer,
                self.comm,
                self.device.stream as *mut _,
            )?;
        }
        Ok(())
    }

    pub fn recv<T: NcclType>(
        &self,
        buff: &mut CudaSlice<T>,
        peer: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::recv(
                buff.cu_device_ptr as *mut _,
                buff.len,
                T::as_nccl_type(),
                peer,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn broadcast<T: NcclType>(
        &self,
        sendbuff: &Option<CudaSlice<T>>,
        recvbuff: &mut CudaSlice<T>,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            let send_ptr = match sendbuff {
                Some(buffer) => buffer.cu_device_ptr as *mut _,
                None => ptr::null(),
            };
            result::broadcast(
                send_ptr,
                recvbuff.cu_device_ptr as *mut _,
                recvbuff.len,
                T::as_nccl_type(),
                root,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn all_gather<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::all_gather(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn all_reduce<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::all_reduce(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn reduce<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::reduce(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                root,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn reduce_scatter<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::reduce_scatter(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                recvbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }
}

#[macro_export]
macro_rules! group {
    ($x:block) => {
        unsafe {
            result::group_start().unwrap();
        }
        $x
        unsafe {
            result::group_end().unwrap();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce() {
        let n = 2;
        let n_devices = CudaDevice::count().unwrap() as usize;
        let id = Id::new().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                println!("III {i}");
                std::thread::spawn(move || {
                    println!("Within thread {i}");
                    let dev = CudaDevice::new(i).unwrap();
                    let comm = Comm::from_rank(dev.clone(), i, n_devices, id).unwrap();
                    let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
                    comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
                        .unwrap();

                    let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

                    assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()
        }
    }
}
