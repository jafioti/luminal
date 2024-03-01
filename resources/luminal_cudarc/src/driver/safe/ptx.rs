use crate::{
    driver::result,
    nvrtc::{Ptx, PtxKind},
};

use super::core::{CudaDevice, CudaModule};

use std::ffi::CString;
use std::{collections::BTreeMap, sync::Arc};

impl CudaDevice {
    /// Dynamically load a set of [crate::driver::CudaFunction] from a jit compiled ptx.
    ///
    /// - `ptx` contains the compilex ptx
    /// - `module_name` is a unique identifier used to access the module later on with [CudaDevice::get_func()]
    /// - `func_names` is a slice of function names to load into the module during build.
    pub fn load_ptx(
        self: &Arc<Self>,
        ptx: Ptx,
        module_name: &str,
        func_names: &[&'static str],
    ) -> Result<(), result::DriverError> {
        self.bind_to_thread()?;

        let cu_module = match ptx.0 {
            PtxKind::Image(image) => unsafe {
                result::module::load_data(image.as_ptr() as *const _)
            },
            PtxKind::Src(src) => {
                let c_src = CString::new(src).unwrap();
                unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
            }
            PtxKind::File(path) => {
                let name_c = CString::new(path.to_str().unwrap()).unwrap();
                result::module::load(name_c)
            }
        }?;
        let mut functions = BTreeMap::new();
        for &fn_name in func_names.iter() {
            let fn_name_c = CString::new(fn_name).unwrap();
            let cu_function = unsafe { result::module::get_function(cu_module, fn_name_c) }?;
            functions.insert(fn_name, cu_function);
        }
        let module = CudaModule {
            cu_module,
            functions,
        };
        #[allow(unused_mut)]
        {
            let mut modules = self.modules.write();
            #[cfg(not(feature = "no-std"))]
            let mut modules = modules.unwrap();
            modules.insert(module_name.into(), module);
        }
        Ok(())
    }
}
