use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashSet},
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
    mem::size_of,
    sync::Arc,
};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use half::f16;
use itertools::Itertools;
use num_traits::FromPrimitive;
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{op::*, prelude::*};

/// Constant value on device
#[derive(Debug, Clone)]
pub struct CudaConstant(Arc<CudaDevice>, f16);
impl PartialEq for CudaConstant {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaConstant {
    fn process(&self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut a = unsafe { self.0.alloc::<f16>(1).unwrap() };
        self.0.htod_copy_into(vec![self.1], &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaContiguous(CudaFunction, Arc<CudaDevice>, ShapeTracker);

impl PartialEq for CudaContiguous {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaContiguous {
    fn new(shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let mut code = format!(
            "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp_a, int numel{}) {{
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
        Self(dev.get_func(&name, &name).unwrap(), dev, shape)
    }
}
impl Operator for CudaContiguous {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let res_shape = tensors[0].1.contiguous();
        let inp_size = res_shape.n_elements();
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

// Unary Op (A -> A)

#[derive(Debug, Clone)]
pub struct CudaLog2(CudaFunction, Arc<CudaDevice>);

impl PartialEq for CudaLog2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaLog2 {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = log2(inp[i]);
    }
}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaLog2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaExp2(CudaFunction, Arc<CudaDevice>);
impl PartialEq for CudaExp2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaExp2 {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = exp2(inp[i]);
    }
}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaExp2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaSin(CudaFunction, Arc<CudaDevice>);
impl PartialEq for CudaSin {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSin {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaSin {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaSqrt(CudaFunction, Arc<CudaDevice>);
impl PartialEq for CudaSqrt {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSqrt {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sqrt(inp[i]);
    }
}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaSqrt {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaRecip(CudaFunction, Arc<CudaDevice>);
impl PartialEq for CudaRecip {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaRecip {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{
        out[i] = hrcp(inp[i]);
    }}
}}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaRecip {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

// Binary Ops

#[derive(Debug, Clone)]
pub struct CudaAdd(CudaFunction, Arc<CudaDevice>, ShapeTracker, ShapeTracker);
impl PartialEq for CudaAdd {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaAdd {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp_a, const __half *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = 
            (({a_valid_exp}) == 0 ? __float2half(0.0) : inp_a[{a_idx_exp}]) 
            + (({b_valid_exp}) == 0 ? __float2half(0.0) : inp_b[{b_idx_exp}]);
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
        Self(dev.get_func(&name, &name).unwrap(), dev, a_shape, b_shape)
    }
}

impl Operator for CudaAdd {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaMul(CudaFunction, Arc<CudaDevice>, ShapeTracker, ShapeTracker);
impl PartialEq for CudaMul {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMul {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp_a, const __half *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = (({a_valid_exp}) == 0 ? __float2half(0.0) : inp_a[{a_idx_exp}]) * (({b_valid_exp}) == 0 ? __float2half(0.0) : inp_b[{b_idx_exp}]);
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
        Self(dev.get_func(&name, &name).unwrap(), dev, a_shape, b_shape)
    }
}

impl Operator for CudaMul {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = unsafe { self.1.alloc::<f16>(inp_size).unwrap() };
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

#[derive(Debug, Clone)]
pub struct CudaMod(CudaFunction, Arc<CudaDevice>, ShapeTracker, ShapeTracker);
impl PartialEq for CudaMod {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMod {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp_a, const __half *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = fmod(
            ({a_valid_exp}) == 0 ? __float2half(0.0) : inp_a[{a_idx_exp}], 
            ({b_valid_exp}) == 0 ? __float2half(0.0) : inp_b[{b_idx_exp}]
        );
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
        Self(dev.get_func(&name, &name).unwrap(), dev, a_shape, b_shape)
    }
}

impl Operator for CudaMod {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaLessThan(CudaFunction, Arc<CudaDevice>, ShapeTracker, ShapeTracker);
impl PartialEq for CudaLessThan {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaLessThan {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp_a, const __half *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        __half a_t = 0.0;
        __half b_t = 0.0;
        if (({a_valid_exp}) != 0) {{
            a_t = inp_a[{a_idx_exp}];
        }}
        if (({b_valid_exp}) != 0) {{
            b_t = inp_b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = __float2half(1.0);
        }} else {{
            out[idx] = __float2half(0.0);
        }}
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
        Self(dev.get_func(&name, &name).unwrap(), dev, a_shape, b_shape)
    }
}

impl Operator for CudaLessThan {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
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

#[derive(Debug, Clone)]
pub struct CudaSumReduce(CudaFunction, Arc<CudaDevice>, pub usize, ShapeTracker);

impl PartialEq for CudaSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSumReduce {
    fn new(dim: usize, shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, const int front_size, const int back_size, const int dim_size, int numel{}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        __half reduce_value = __float2half(0.0);
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value = reduce_value + inp[{idx_exp}];
            }}
        }}
        out[i_] = reduce_value;
    }}
}}",
            shape
                .dims
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
        Self(dev.get_func(&name, &name).unwrap(), dev, dim, shape)
    }
}
impl Operator for CudaSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.2);
        let inp_size = shape.n_elements();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
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

        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut added = HashSet::new();
        let mut dims = [0; 10];
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[0].1.shape()) {
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

#[derive(Debug, Clone)]
pub struct CudaMaxReduce(CudaFunction, Arc<CudaDevice>, pub usize, ShapeTracker);

impl PartialEq for CudaMaxReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMaxReduce {
    fn new(dim: usize, shape: ShapeTracker, dev: Arc<CudaDevice>) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let mut code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, const int front_size, const int back_size, const int dim_size, int numel{}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        __half reduce_value = __float2half(-__int_as_float(0x7f800000));
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                int a_idx = {idx_exp};
                reduce_value = __hmax(reduce_value, inp[a_idx]);
            }}
        }}
        out[i_] = reduce_value;
    }}
}}",
            shape
                .dims
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
        Self(dev.get_func(&name, &name).unwrap(), dev, dim, shape)
    }
}
impl Operator for CudaMaxReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.2);
        let inp_size = shape.n_elements();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
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

        let out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut added = HashSet::new();
        let mut dims = [0; 10];
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[0].1.shape()) {
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

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct CudaPrimitiveCompiler;

impl Compiler for CudaPrimitiveCompiler {
    fn compile(&self, graph: &mut Graph) {
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
            if graph
                .graph
                .node_weight(function_node)
                .unwrap()
                .as_any()
                .downcast_ref::<Function>()
                .unwrap()
                .2
                == std::any::TypeId::of::<Vec<f32>>()
            {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyToDevice::<f16>::new(dev.clone()))
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

                move_references(
                    &mut graph.id_remap,
                    &mut graph.no_delete,
                    &mut graph.to_retrieve,
                    function_node,
                    copy_node,
                );
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
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
            // Filter non-functions
            .filter(|n| {
                !graph
                    .graph
                    .node_weight(**n)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
            })
            .map(|n| {
                (
                    *n,
                    graph
                        .graph
                        .edges_directed(*n, petgraph::Direction::Incoming)
                        .flat_map(|i| i.weight().as_data().map(|i| i.2))
                        .max_by_key(|s| s.n_physical_elements())
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
                .input(output_node, 0, output_shape)
                .finish();

            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                output_node,
                copy_node,
            );
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
                        .find(|i| !i.weight().is_schedule())
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
                .add_op(CudaCopyFromDevice::<f16>::new(dev.clone()))
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(
                copy_node,
                output_node,
                Dependency::Data {
                    input_order: 0,
                    output_order: 0,
                    shape,
                },
            );
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::new(dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::new(dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::new(dev.clone()));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::new(dev.clone()));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant(dev.clone(), f16::from_f32(c.0)));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::new(dev.clone()));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::new(src_shapes[0], src_shapes[1], dev.clone()));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::new(src_shapes[0], src_shapes[1], dev.clone()));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::new(src_shapes[0], src_shapes[1], dev.clone()));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::new(src_shapes[0], src_shapes[1], dev.clone()));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::new(src_shapes[0], dev.clone()));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::new(*dim, src_shapes[0], dev.clone()));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::new(*dim, src_shapes[0], dev.clone()));
            }
        }
    }
}

/// In 16 bit, summing above 2048 doesn't work. This precludes the .expand(Dim).sum_reduce() pattern to get a dim size in a tensor, so we need to replace these fake reductions with an elementwise mul
#[derive(Debug, Default)]
pub struct FakeReductionCompiler;

impl Compiler for FakeReductionCompiler {
    fn compile(&self, graph: &mut Graph) {
        let mut sum_reduce = NodeIndex::default();
        let s = SelectEdge::new(
            SelectOp::new().check(|o, _| {
                if let Some(c) = o.as_any().downcast_ref::<CudaConstant>() {
                    c.1 == f16::ONE
                } else {
                    false
                }
            }),
            SelectOp::new()
                .ty::<CudaSumReduce>()
                .check(|o, shapes| {
                    if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce>() {
                        shapes[0].fake[shapes[0].indexes[o.2]] // Ensure dimension we are reducing is fake
                    } else {
                        false
                    }
                })
                .ptr(&mut sum_reduce),
        );

        for _ in s.search(graph) {
            let op_ref = graph.graph.node_weight_mut(sum_reduce).unwrap();
            let dim = op_ref.as_any().downcast_ref::<CudaSumReduce>().unwrap().2;
            let dev = op_ref
                .as_any()
                .downcast_ref::<CudaSumReduce>()
                .unwrap()
                .1
                .clone();
            *op_ref = Box::new(FakeSumReduce::new(dev, dim));
        }
    }
}

#[derive(Debug, Clone)]
pub struct FakeSumReduce(CudaFunction, Arc<CudaDevice>, pub usize);
impl PartialEq for FakeSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl FakeSumReduce {
    pub fn new(dev: Arc<CudaDevice>, dim: usize) -> Self {
        let mut code = "#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(__half *out, const __half *inp, int numel, __half mul_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = inp[i] * mul_factor;
    }
}"
        .to_string();
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
        Self(dev.get_func(&name, &name).unwrap(), dev, dim)
    }
}

impl Operator for FakeSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dim_size = f16::from_usize(tensors[0].1.shape()[self.2].to_usize().unwrap()).unwrap();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = self.1.alloc_zeros::<f16>(inp_size).unwrap();
        unsafe {
            self.0
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(inp_size as u32),
                    (&mut out, inp, inp_size, dim_size),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler;

impl Compiler for CopyCompiler {
    fn compile(&self, graph: &mut Graph) {
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
                    .is::<CudaCopyToDevice<f16>>()
                    && graph
                        .graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<f16>>())
                    || (graph
                        .graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<f16>>()
                        && graph
                            .graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<f16>>())
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
                        .is::<CudaCopyFromDevice<f16>>()
                        && !graph
                            .graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<f16>>()
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
                &mut graph.id_remap,
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
                    &mut graph.id_remap,
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

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
