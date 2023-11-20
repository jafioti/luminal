mod fp16;
mod fp32;

pub use fp16::CudaFp16Compiler;
pub use fp32::CudaFp32Compiler;
use half::f16;
use itertools::Itertools;

use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    fmt::{Debug, Write},
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
#[derive(Clone, LuminalEq, LuminalPrint)]
pub struct CudaCopyToDevice<T>(Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaCopyToDevice<T> {
    fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyToDevice(dev, Default::default())
    }
}

impl<T> Operator for CudaCopyToDevice<T>
where
    CudaSlice<T>: Data,
    T: ConvertF32 + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
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
            .map(ConvertF32::from)
            .collect::<Vec<_>>();
        let mut a = unsafe { self.0.alloc::<T>(vec.len()).unwrap() };
        self.0.htod_copy_into(vec, &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone, LuminalEq, LuminalPrint)]
pub struct CudaCopyFromDevice<T>(Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaCopyFromDevice<T> {
    fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyFromDevice(dev, Default::default())
    }
}

impl<T> Operator for CudaCopyFromDevice<T>
where
    CudaSlice<T>: Data,
    T: ConvertF32 + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
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
                    .map(ConvertF32::to)
                    .collect::<Vec<_>>(),
            ),
        }]
    }
}

pub trait ConvertF32 {
    fn to(self) -> f32;
    fn from(a: f32) -> Self;
}

impl ConvertF32 for f32 {
    fn from(a: f32) -> Self {
        a
    }
    fn to(self) -> f32 {
        self
    }
}

impl ConvertF32 for f16 {
    fn from(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to(self) -> f32 {
        self.to_f32()
    }
}

/// Constant value on device
#[derive(LuminalPrint, Clone, LuminalEq)]
pub struct CudaConstant<T>(Arc<CudaDevice>, T);

impl<T> CudaConstant<T> {
    fn new(dev: Arc<CudaDevice>, val: T) -> Self {
        Self(dev, val)
    }
}

impl<T> Operator for CudaConstant<T>
where
    T: Debug + Copy + cudarc::driver::DeviceRepr + std::marker::Unpin,
    CudaSlice<T>: Data,
{
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut a = unsafe { self.0.alloc::<T>(1).unwrap() };
        self.0.htod_copy_into(vec![self.1], &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

#[derive(LuminalPrint, Clone, LuminalEq)]
pub struct CudaContiguous<T>(CudaFunction, Arc<CudaDevice>, ShapeTracker, PhantomData<T>);

impl<T> CudaContiguous<T> {
    fn new(shape: ShapeTracker, dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel && ({valid_exp}) != 0) {{
        out[idx] = inp_a[{idx_exp}];
    }}
}}",
            shape
                .shape()
                .into_iter()
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .fold(String::default(), |mut acc, c| {
                    write!(&mut acc, ", int {c}").unwrap();
                    acc
                })
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
        )
    }
}
impl<T> Operator for CudaContiguous<T>
where
    T: Debug + 'static + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    CudaSlice<T>: Data,
{
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let res_shape = tensors[0].1.contiguous();
        let inp_size = res_shape.n_elements();
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
        let mut added = HashSet::new();
        let mut dims = [0; 10];
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    dims[added.len()] = d2.to_usize().unwrap() as i32;
                    added.insert(c);
                }
            }
        }
        for i in 0..added.len() {
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaLog2<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaLog2<T> {
    pub fn new(dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = log2(inp[i]);
    }}
}}"
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaExp2<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaExp2<T> {
    pub fn new(dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = exp2(inp[i]);
    }}
}}"
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaSin<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaSin<T> {
    pub fn new(dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let mut code = format!(
            "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = sin(inp[i]);
    }}
}}"
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaSqrt<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaSqrt<T> {
    pub fn new(dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = {}(inp[i]);
    }}
}}",
            if type_name == "float" {
                "sqrt"
            } else {
                "hsqrt"
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaRecip<T>(CudaFunction, Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaRecip<T> {
    pub fn new(dev: Arc<CudaDevice>, type_name: &str) -> Self {
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = {}(inp[i]);
    }}
}}",
            if type_name == "float" {
                "__frcp_rn"
            } else {
                "hrcp"
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
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

#[derive(LuminalEq, LuminalPrint, Clone)]
pub struct CudaAdd<T>(
    CudaFunction,
    Arc<CudaDevice>,
    ShapeTracker,
    ShapeTracker,
    PhantomData<T>,
);

impl<T> CudaAdd<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        dev: Arc<CudaDevice>,
        type_name: &str,
    ) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? {} : inp_a[{a_idx_exp}]) 
            + (({b_valid_exp}) == 0 ? {} : inp_b[{b_idx_exp}]);
    }}
}}",
            a_shape
                .shape()
                .into_iter()
                .chain(b_shape.shape().into_iter())
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .fold(String::default(), |mut acc, c| {
                    write!(&mut acc, ", int {c}").unwrap();
                    acc
                }),
                if type_name == "float" {"0.0"} else {"__float2half(0.0)"},
                if type_name == "float" {"0.0"} else {"__float2half(0.0)"},
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
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
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
        let inp_size: usize = tensors[0].1.n_elements();

        let out = self.1.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut added = HashSet::new();
        let mut dims = [0; 10];
        for (d1, d2) in self
            .2
            .shape()
            .into_iter()
            .zip(tensors[0].1.shape())
            .chain(self.3.shape().into_iter().zip(tensors[1].1.shape()))
        {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    dims[added.len()] = d2.to_usize().unwrap() as i32;
                    added.insert(c);
                }
            }
        }
        for i in 0..added.len() {
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
