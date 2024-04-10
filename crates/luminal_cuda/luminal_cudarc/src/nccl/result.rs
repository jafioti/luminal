//! A thin wrapper around [sys] providing [Result]s with [NcclError].

use super::sys::{self, ncclCommSplit, ncclGetVersion, ncclRedOpCreatePreMulSum, ncclRedOpDestroy};
use std::mem::MaybeUninit;

/// Wrapper around [sys::ncclResult_t].
/// See [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html?ncclresult-t)
#[derive(Clone, PartialEq, Eq)]
pub struct NcclError(pub sys::ncclResult_t);

impl std::fmt::Debug for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NcclError")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum NcclStatus {
    Success,
    InProgress,
    NumResults,
}

impl sys::ncclResult_t {
    /// Transforms into a [Result] of [NcclError]
    pub fn result(self) -> Result<NcclStatus, NcclError> {
        match self {
            sys::ncclResult_t::ncclSuccess => Ok(NcclStatus::Success),
            sys::ncclResult_t::ncclInProgress => Ok(NcclStatus::InProgress),
            sys::ncclResult_t::ncclNumResults => Ok(NcclStatus::NumResults),
            _ => Err(NcclError(self)),
        }
    }
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?c.ncclCommFinalize)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_finalize(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    sys::ncclCommFinalize(comm).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommdestroy)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_destroy(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    sys::ncclCommDestroy(comm).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommabort)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_abort(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    sys::ncclCommAbort(comm).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommcount)
pub fn get_nccl_version() -> Result<::core::ffi::c_int, NcclError> {
    let mut version: ::core::ffi::c_int = 0;
    unsafe {
        ncclGetVersion(&mut version).result()?;
    }
    Ok(version)
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclgetuniqueid)
pub fn get_uniqueid() -> Result<sys::ncclUniqueId, NcclError> {
    let mut uniqueid = MaybeUninit::uninit();
    Ok(unsafe {
        sys::ncclGetUniqueId(uniqueid.as_mut_ptr()).result()?;
        uniqueid.assume_init()
    })
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcomminitrankconfig)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_init_rank_config(
    comm: *mut sys::ncclComm_t,
    nranks: ::core::ffi::c_int,
    comm_id: sys::ncclUniqueId,
    rank: ::core::ffi::c_int,
    config: *mut sys::ncclConfig_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclCommInitRankConfig(comm, nranks, comm_id, rank, config).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcomminitrank)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_init_rank(
    comm: *mut sys::ncclComm_t,
    nranks: ::core::ffi::c_int,
    comm_id: sys::ncclUniqueId,
    rank: ::core::ffi::c_int,
) -> Result<NcclStatus, NcclError> {
    sys::ncclCommInitRank(comm, nranks, comm_id, rank).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcomminitall)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_init_all(
    comm: *mut sys::ncclComm_t,
    ndev: ::core::ffi::c_int,
    devlist: *const ::core::ffi::c_int,
) -> Result<NcclStatus, NcclError> {
    sys::ncclCommInitAll(comm, ndev, devlist).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommsplit)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_split(
    comm: sys::ncclComm_t,
    color: ::core::ffi::c_int,
    key: ::core::ffi::c_int,
    newcomm: *mut sys::ncclComm_t,
    config: *mut sys::ncclConfig_t,
) -> Result<NcclStatus, NcclError> {
    ncclCommSplit(comm, color, key, newcomm, config).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommcount)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_count(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    let mut count = 0;
    sys::ncclCommCount(comm, &mut count).result()?;
    Ok(count)
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommcudevice)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_cu_device(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    let mut device = 0;
    sys::ncclCommCuDevice(comm, &mut device).result()?;
    Ok(device)
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcommuserrank)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn comm_user_rank(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    let mut rank = 0;
    sys::ncclCommUserRank(comm, &mut rank).result()?;
    Ok(rank)
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html?c.ncclRedOpCreatePreMulSum)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn reduce_op_create_pre_mul_sum(
    op: *mut sys::ncclRedOp_t,
    scalar: *mut ::core::ffi::c_void,
    datatype: sys::ncclDataType_t,
    residence: sys::ncclScalarResidence_t,
    comm: sys::ncclComm_t,
) -> Result<NcclStatus, NcclError> {
    ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html?ncclredopdestroy)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn reduce_op_destroy(
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
) -> Result<NcclStatus, NcclError> {
    ncclRedOpDestroy(op, comm).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html?ncclreduce)
/// # Safety
/// User is in charge of sending valid pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    root: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html?ncclbroadcast)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn broadcast(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    root: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html?ncclallreduce)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn all_reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html?ncclreducescatter)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn reduce_scatter(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    recvcount: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html?ncclallgather)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn all_gather(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    sendcount: usize,
    datatype: sys::ncclDataType_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html?ncclsend)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn send(
    sendbuff: *const ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclSend(sendbuff, count, datatype, peer, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html?ncclrecv)
/// # Safety
/// User is in charge of sending valid pointers.
pub unsafe fn recv(
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    sys::ncclRecv(recvbuff, count, datatype, peer, comm, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html?c.ncclGroupEnd)
pub fn group_end() -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclGroupEnd().result() }
}

/// See [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html?ncclgroupstart)
pub fn group_start() -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclGroupStart().result() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CudaDevice;
    use std::ffi::c_void;

    #[test]
    fn single_thread() {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let n = 2;

        let mut devs = vec![];
        let mut sendslices = vec![];
        let mut recvslices = vec![];
        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
            sendslices.push(slice);
            let slice = dev.alloc_zeros::<f32>(n).unwrap();
            recvslices.push(slice);
            devs.push(dev);
        }
        let mut comms = vec![std::ptr::null_mut(); n_devices];
        let ordinals: Vec<_> = devs.iter().map(|d| d.ordinal as i32).collect();
        unsafe {
            comm_init_all(comms.as_mut_ptr(), n_devices as i32, ordinals.as_ptr()).unwrap();

            group_start().unwrap();
            for i in 0..n_devices {
                // Very important to set the cuda context to this device.
                let dev = CudaDevice::new(i).unwrap();
                all_reduce(
                    sendslices[i].cu_device_ptr as *const c_void,
                    recvslices[i].cu_device_ptr as *mut c_void,
                    n,
                    sys::ncclDataType_t::ncclFloat32,
                    sys::ncclRedOp_t::ncclSum,
                    comms[i],
                    dev.stream as sys::cudaStream_t,
                )
                .unwrap();
            }
            group_end().unwrap();
        }
        for (i, recv) in recvslices.iter().enumerate() {
            // Get the current device context
            let dev = CudaDevice::new(i).unwrap();
            let out = dev.dtoh_sync_copy(recv).unwrap();
            assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
        }
    }

    #[test]
    fn multi_thread() {
        let n_devices = CudaDevice::count().unwrap() as usize;

        let n = 2;
        let comm_id = get_uniqueid().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                let n_devices = n_devices.clone();
                std::thread::spawn(move || {
                    let dev = CudaDevice::new(i).unwrap();
                    let sendslice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let recvslice = dev.alloc_zeros::<f32>(n).unwrap();
                    let mut comm = MaybeUninit::uninit();
                    unsafe {
                        comm_init_rank(comm.as_mut_ptr(), n_devices as i32, comm_id, i as i32)
                            .unwrap();

                        let comm = comm.assume_init();
                        use std::ffi::c_void;
                        all_reduce(
                            sendslice.cu_device_ptr as *const c_void,
                            recvslice.cu_device_ptr as *mut c_void,
                            n,
                            sys::ncclDataType_t::ncclFloat32,
                            sys::ncclRedOp_t::ncclSum,
                            comm,
                            dev.stream as sys::cudaStream_t,
                        )
                        .unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }
}
