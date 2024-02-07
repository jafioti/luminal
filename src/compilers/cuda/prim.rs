use super::{get_idx_valid_exps, render_dyn_dim_inputs};
use rustc_hash::FxHashMap;

use std::{
    collections::hash_map::DefaultHasher,
    fmt::Debug,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::size_of,
    sync::Arc,
};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use crate::{op::*, prelude::*};

/// Copy a tensor to the GPU
#[derive(Clone, LuminalEqFalse, LuminalPrint)]
pub struct CudaCopyToDevice<T>(Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaCopyToDevice<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyToDevice(dev, Default::default())
    }
}

impl<T> Operator for CudaCopyToDevice<T>
where
    CudaSlice<T>: Data,
    T: CudaFloat + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<CudaSlice<T>>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cpu_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let vec = cpu_data
            .iter()
            .copied()
            .map(CudaFloat::from_f32)
            .collect::<Vec<_>>();
        let mut a = unsafe { self.0.alloc::<T>(vec.len()).unwrap() };
        self.0.htod_copy_into(vec, &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone, LuminalEqFalse, LuminalPrint)]
pub struct CudaCopyFromDevice<T>(Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaCopyFromDevice<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyFromDevice(dev, Default::default())
    }
}

impl<T> Operator for CudaCopyFromDevice<T>
where
    CudaSlice<T>: Data,
    T: CudaFloat + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cuda_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        vec![Tensor {
            data: Box::new(
                self.0
                    .dtoh_sync_copy(cuda_data)
                    .unwrap()
                    .into_iter()
                    .map(CudaFloat::to_f32)
                    .collect::<Vec<_>>(),
            ),
        }]
    }
}

/// Constant value on device
#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct CudaConstant<T>(
    pub ConstantValue,
    Arc<CudaDevice>,
    *const FxHashMap<char, usize>,
    PhantomData<T>,
);

impl<T> CudaConstant<T> {
    pub fn new(
        dev: Arc<CudaDevice>,
        val: ConstantValue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        Self(val, dev, dyn_map, Default::default())
    }
}

impl<T> Operator for CudaConstant<T>
where
    T: Debug + Copy + cudarc::driver::DeviceRepr + std::marker::Unpin + CudaFloat,
    CudaSlice<T>: Data,
{
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut a = unsafe { self.1.alloc::<T>(1).unwrap() };
        let value = match &self.0 {
            ConstantValue::Expression(e) => {
                T::from_f32(e.exec(unsafe { self.2.as_ref().unwrap() }).unwrap() as f32)
            }
            ConstantValue::Float(f) => T::from_f32(*f),
        };
        self.1.htod_copy_into(vec![value], &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct CudaContiguous<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaContiguous<T> {
    pub fn new(
        shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);

        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp_a, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && ({valid}) != 0) {{
        out[idx] = inp_a[{idx}];
    }}
}}",
            T::type_name(),
            T::type_name(),
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}
impl<T> Operator for CudaContiguous<T>
where
    T: Debug + 'static + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let res_shape = tensors[0].1.contiguous();
        let inp_size = res_shape.n_elements().to_usize().unwrap();
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.5.as_ref().unwrap() };
        for (i, d) in self.4.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaLog2<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat> CudaLog2<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = log2(inp[i]);
    }}
}}",
            T::type_name(),
            T::type_name()
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, Default::default())
    }
}

impl<T> Operator for CudaLog2<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaExp2<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat> CudaExp2<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = exp2(inp[i]);
    }}
}}",
            T::type_name(),
            T::type_name()
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, Default::default())
    }
}

impl<T> Operator for CudaExp2<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaSin<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat> CudaSin<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = format!(
            "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = sin(inp[i]);
    }}
}}",
            T::type_name(),
            T::type_name()
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, Default::default())
    }
}

impl<T> Operator for CudaSin<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaSqrt<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat> CudaSqrt<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = {}(inp[i]);
    }}
}}",
            T::type_name(),
            T::type_name(),
            if T::is_f32() { "sqrt" } else { "hsqrt" }
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, Default::default())
    }
}

impl<T> Operator for CudaSqrt<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaRecip<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T: CudaFloat> CudaRecip<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = {}(inp[i]);
    }}
}}",
            T::type_name(),
            T::type_name(),
            if T::is_f32() { "__frcp_rn" } else { "hrcp" }
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(dev.get_func(&name, &name).unwrap(), dev, Default::default())
    }
}

impl<T> Operator for CudaRecip<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaAdd<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaAdd<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp_a, const {} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = 
            (({a_valid}) == 0 ? {} : inp_a[{a_idx}]) 
            + (({b_valid}) == 0 ? {} : inp_b[{b_idx}]);
    }}
}}",
            T::type_name(),
            T::type_name(),
            T::type_name(),
            if T::is_f32() {
                "0.0"
            } else {
                "__float2half(0.0)"
            },
            if T::is_f32() {
                "0.0"
            } else {
                "__float2half(0.0)"
            },
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}

impl<T> Operator for CudaAdd<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaMul<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaMul<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp_a, const {} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = (({a_valid}) == 0 ? {} : inp_a[{a_idx}]) * (({b_valid}) == 0 ? {} : inp_b[{b_idx}]);
    }}
}}", T::type_name(), T::type_name(), T::type_name(), if T::is_f32() {"0.0"} else {"__float2half(0.0)"}, if T::is_f32() {"0.0"} else {"__float2half(0.0)"}
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}

impl<T> Operator for CudaMul<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = unsafe { self.1.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaMod<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaMod<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let mut code = format!(
            "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp_a, const {} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = fmod(
            ({a_valid}) == 0 ? {} : inp_a[{a_idx}], 
            ({b_valid}) == 0 ? {} : inp_b[{b_idx}]
        );
    }}
}}",
            T::type_name(),
            T::type_name(),
            T::type_name(),
            if T::is_f32() {
                "0.0"
            } else {
                "__float2half(0.0)"
            },
            if T::is_f32() {
                "0.0"
            } else {
                "__float2half(0.0)"
            }
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}

impl<T> Operator for CudaMod<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaLessThan<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaLessThan<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        {type_name} a_t = 0.0;
        {type_name} b_t = 0.0;
        if (({a_valid}) != 0) {{
            a_t = inp_a[{a_idx}];
        }}
        if (({b_valid}) != 0) {{
            b_t = inp_b[{b_idx}];
        }}
        if (a_t < b_t) {{
            out[idx] = {};
        }} else {{
            out[idx] = {};
        }}
    }}
}}",
                if T::is_f32() {"1.0"} else {"__float2half(1.0)"},
                if T::is_f32() {"0.0"} else {"__float2half(0.0)"}
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            a_shape,
            b_shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}

impl<T> Operator for CudaLessThan<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaSumReduce<T>(
    CudaFunction,
    pub Arc<CudaDevice>,
    pub usize,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaSumReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        {} reduce_value = {};
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = reduce_value + inp[{idx}];
            }}
        }}
        out[i_] = reduce_value;
    }}
}}", T::type_name(), T::type_name(), T::type_name(), if T::is_f32() {"0.0"} else {"__float2half(0.0)"}
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            dim,
            shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}
impl<T> Operator for CudaSumReduce<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.2);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.2)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.2 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.2].to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }
        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct CudaMaxReduce<T>(
    CudaFunction,
    Arc<CudaDevice>,
    pub usize,
    ShapeTracker,
    PhantomData<T>,
    Vec<char>,
    *const FxHashMap<char, usize>,
);

impl<T: CudaFloat> CudaMaxReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} *out, const {} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        {} reduce_value = {}(-__int_as_float(0x7f800000));
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                int a_idx = {idx};
                reduce_value = {}(reduce_value, inp[a_idx]);
            }}
        }}
        out[i_] = reduce_value;
    }}
}}", T::type_name(), T::type_name(), T::type_name(), if T::is_f32() {""} else {"__float2half"}, if T::is_f32() {"max"} else {"__hmax"}
        );
        let name = format!("kernel_{}", hash(&code));
        code = code.replace("kernel", &name);
        if !dev.has_func(&name, &name) {
            dev.load_ptx(
                compile_ptx_with_opts(
                    code,
                    CompileOptions {
                        arch: Some("sm_75"),
                        include_paths: vec!["/usr/local/cuda/include".to_string()],
                        ..Default::default()
                    },
                )
                .unwrap(),
                &name,
                &[&name],
            )
            .unwrap();
        }
        Self(
            dev.get_func(&name, &name).unwrap(),
            dev,
            dim,
            shape,
            Default::default(),
            dyn_symbols,
            dyn_map,
        )
    }
}
impl<T> Operator for CudaMaxReduce<T>
where
    T: Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.2);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.2)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.2 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.2].to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.6.as_ref().unwrap() };
        for (i, d) in self.5.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe { dims[0].as_kernel_param().add(i * size_of::<i32>()) });
        }
        unsafe {
            self.0
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
