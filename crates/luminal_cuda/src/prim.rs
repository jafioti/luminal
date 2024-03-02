use crate::{hash, CudaData, CudaFloat};

use super::{get_idx_valid_exps, render_dyn_dim_inputs};
use itertools::Itertools;
use rustc_hash::FxHashMap;

use std::{
    any::{Any, TypeId},
    fmt::Debug,
    marker::PhantomData,
    mem::size_of,
    sync::Arc,
};

use luminal_cudarc::{
    driver::{CudaDevice, CudaFunction, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use luminal::{
    op::{Function as LFunction, *},
    prelude::{petgraph::visit::EdgeRef, *},
};

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
    CudaData<T>: Data,
    T: CudaFloat + luminal_cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<CudaData<T>>() {
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
        vec![Tensor {
            data: Box::new(CudaData(a)),
        }]
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
    CudaData<T>: Data,
    T: CudaFloat + luminal_cudarc::driver::DeviceRepr + std::marker::Unpin,
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
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        vec![Tensor {
            data: Box::new(
                self.0
                    .dtoh_sync_copy(&cuda_data.0)
                    .unwrap()
                    .into_iter()
                    .map(CudaFloat::to_f32)
                    .collect::<Vec<_>>(),
            ),
        }]
    }
}

/// Constant value on device
#[derive(Clone, LuminalEqFalse)]
pub struct CudaConstant<T>(
    pub ConstantValue,
    Arc<CudaDevice>,
    *const FxHashMap<char, usize>,
    PhantomData<T>,
);
impl<T> Debug for CudaConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaConstant({:?})", self.0)
    }
}

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
    T: Debug + Copy + luminal_cudarc::driver::DeviceRepr + std::marker::Unpin + CudaFloat,
    CudaData<T>: Data,
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
        vec![Tensor {
            data: Box::new(CudaData(a)),
        }]
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
                &[name.clone().leak()],
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
    T: Debug
        + 'static
        + luminal_cudarc::driver::DeviceRepr
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let res_shape = tensors[0].1.contiguous();
        let inp_size = res_shape.n_elements().to_usize().unwrap();
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, &(inp.0), inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, &(inp.0), inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, &(inp.0), inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, &(inp.0), inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements().to_usize().unwrap();
        let mut out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, &(inp.0), inp_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = unsafe { self.1.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
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
        let type_name = T::type_name();
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = reduce_value + (float)inp[{idx}];
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
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
                &[name.clone().leak()],
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
    T: CudaFloat,
    CudaData<T>: Data,
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
            .downcast_ref::<CudaData<T>>()
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
            (&inp.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }
        vec![Tensor {
            data: Box::new(CudaData(out)),
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
        let type_name = T::type_name();
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = -__int_as_float(0x7f800000);
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                int a_idx = {idx};
                reduce_value = max(reduce_value, (float)inp[a_idx]);
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
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
                &[name.clone().leak()],
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
        + luminal_cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + luminal_cudarc::driver::ValidAsZeroBits,
    CudaData<T>: Data,
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
            .downcast_ref::<CudaData<T>>()
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
            (&inp.0).as_kernel_param(),
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
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(LuminalPrint, Default)]
pub struct CudaPrimitiveCompiler<T>(PhantomData<T>);

impl<T: CudaFloat + 'static> Compiler for CudaPrimitiveCompiler<T>
where
    CudaData<T>: Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let dev = CudaDevice::new(0).unwrap();
        // Go through the graph and insert copy ops
        // Copy function output to device and input from device
        for function_node in graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .node_weight(*n)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
                    && graph.graph.edges(*n).count() != 0
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice::<T>::new(dev.clone()))
                .input(function_node, 0, ShapeTracker::new(&[]))
                .finish();

            // Switch outgoing edges from input to copy_node
            for (edge_id, weight, dest) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Outgoing)
                .map(|e| (e.id(), *e.weight(), e.target()))
                .filter(|(_, _, trg)| *trg != copy_node)
                .collect::<Vec<_>>()
            {
                graph.graph.add_edge(copy_node, dest, weight);
                graph.graph.remove_edge(edge_id);
            }

            if graph.to_retrieve.contains(&function_node) {
                graph.to_retrieve.insert(copy_node);
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(source, 0, ShapeTracker::new(&[]))
                    .finish();
                graph
                    .graph
                    .add_edge(copy_from_node, function_node, edge_weight);
                graph.graph.remove_edge(edge);
            }
        }

        // Copy to_retrieve from device
        for (output_node, output_shape) in graph
            .to_retrieve
            .iter()
            // Filter to non-functions
            .filter(|n| {
                !graph
                    .graph
                    .node_weight(**n)
                    .unwrap()
                    .as_any()
                    .is::<LFunction>()
            })
            .map(|n| {
                (
                    *n,
                    graph
                        .graph
                        .edges_directed(*n, petgraph::Direction::Incoming)
                        .filter_map(|e| e.weight().as_data())
                        .map(|i| i.2)
                        .max_by_key(|s| s.n_physical_elements().to_usize().unwrap_or_default())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<CudaCopyToDevice<T>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                graph.no_delete.remove(&output_node);
                graph.to_retrieve.remove(&output_node);
                graph.no_delete.insert(src);
                graph.to_retrieve.insert(src);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(output_node, 0, output_shape)
                    .finish();

                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    output_node,
                    copy_node,
                );
            }
        }

        // Copy prints from device
        for (output_node, edge) in graph
            .graph
            .node_indices()
            // Filter non-functions
            .filter(|n| graph.graph.node_weight(*n).unwrap().as_any().is::<Print>())
            .map(|n| {
                (
                    n,
                    graph
                        .graph
                        .edges_directed(n, petgraph::Direction::Incoming)
                        .find(|e| !e.weight().is_schedule())
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().as_data().unwrap().2,
            );
            let copy_node = graph
                .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(
                copy_node,
                output_node,
                Dependency::Data {
                    shape,
                    input_order: 0,
                    output_order: 0,
                },
            );
            graph.graph.remove_edge(edge);
        }

        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::<T>::new(dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::<T>::new(dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::<T>::new(dev.clone()));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant::<T>::new(
                    dev.clone(),
                    c.0.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::<T>::new(dev.clone()));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::<T>::new(dev.clone()));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::<T>::new(
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            }
        }
    }
}

// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: 'static> Compiler for CopyCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        for (first, second) in graph
            .graph
            .edge_indices()
            .filter_map(|e| graph.graph.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph
                    .graph
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<CudaCopyToDevice<T>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>()
        {
            if graph
                .graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph.graph.contains_node(e.target()))
                .filter(|e| {
                    !graph
                        .graph
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let source = graph.get_sources(first)[0];
            move_outgoing_edge(second, source.0, &mut graph.graph);
            move_references(
                &mut remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                second,
                source.0,
            );
            graph.graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph.graph);
                move_references(
                    &mut remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    dest,
                    source.0,
                );
                graph.graph.remove_node(dest);
            }
            graph.graph.remove_node(first);
        }
    }
}
