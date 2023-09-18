//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaDevice], [CudaStream], and more.

pub(crate) mod alloc;
pub(crate) mod core;
pub(crate) mod device_ptr;
pub(crate) mod external_memory;
pub(crate) mod launch;
pub(crate) mod profile;
pub(crate) mod ptx;
pub(crate) mod threading;

pub use self::alloc::{DeviceRepr, ValidAsZeroBits};
pub use self::core::{CudaDevice, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut};
pub use self::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};
pub use self::external_memory::{ExternalMemory, MappedBuffer};
pub use self::launch::{LaunchAsync, LaunchConfig};
pub use self::profile::{profiler_start, profiler_stop, Profiler};

pub use crate::driver::result::DriverError;
