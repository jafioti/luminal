use std::{
    any::{Any, TypeId},
    collections::HashSet,
    sync::Arc,
};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use itertools::Itertools;
use petgraph::visit::EdgeRef;

use crate::{op::*, prelude::*};

/// Copy a tensor to the GPU
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCopyToDevice;

impl Operator for CudaCopyToDevice {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dev = CudaDevice::new(0).unwrap();
        let cpu_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let mut a: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
        dev.htod_sync_copy_into(cpu_data, &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

/// Copy a tensor from the GPU
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCopyFromDevice;

impl Operator for CudaCopyFromDevice {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dev = CudaDevice::new(0).unwrap();
        let cuda_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let a = dev.dtoh_sync_copy(cuda_data).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaContiguous(Arc<CudaDevice>, CudaFunction, ShapeTracker);

impl PartialEq for CudaContiguous {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaContiguous {
    fn new(shape: ShapeTracker) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void contiguous_kernel(float *out, const float *inp_a, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {idx_exp};
    if (idx < numel && {valid_exp} != 0) {{
        out[idx] = inp_a[a_idx];
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "contiguous", &["contiguous_kernel"])
            .unwrap();
        let f = dev.get_func("contiguous", "contiguous_kernel").unwrap();
        Self(dev, f, shape)
    }
}
impl Operator for CudaContiguous {
    fn process(&self, mut tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if tensors[0].1.is_contiguous() {
            // It's already contiguous
            return vec![tensors.pop().unwrap().0.cloned()];
        }
        let res_shape = tensors[0].1.contiguous();
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape
            .shape()
            .iter()
            .map(|d| d.to_usize().unwrap())
            .product();

        let out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            (inp_size as i32).as_kernel_param(),
        ];
        let mut added = HashSet::new();
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added.insert(c);
                }
            }
        }
        unsafe { self.1.clone().launch_async_impl(cfg, &mut params) }.unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

// Unary Op (A -> A)

#[derive(Debug, Clone)]
pub struct CudaLog2(Arc<CudaDevice>, CudaFunction);

impl PartialEq for CudaLog2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaLog2 {
    pub fn new() -> Self {
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void log2_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = log2(inp[i]);
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "log2", &["log2_kernel"]).unwrap();
        let f = dev.get_func("log2", "log2_kernel").unwrap();
        Self(dev, f)
    }
}

impl Operator for CudaLog2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        unsafe {
            self.1.clone().launch(
                LaunchConfig::for_num_elems(inp_size as u32),
                (&mut out, inp, inp_size as i32),
            )
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaExp2(Arc<CudaDevice>, CudaFunction);
impl PartialEq for CudaExp2 {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaExp2 {
    pub fn new() -> Self {
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void exp2_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = exp2(inp[i]);
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "exp2", &["exp2_kernel"]).unwrap();
        let f = dev.get_func("exp2", "exp2_kernel").unwrap();
        Self(dev, f)
    }
}

impl Operator for CudaExp2 {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        unsafe {
            self.1.clone().launch(
                LaunchConfig::for_num_elems(inp_size as u32),
                (&mut out, inp, inp_size as i32),
            )
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaSin(Arc<CudaDevice>, CudaFunction);
impl PartialEq for CudaSin {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSin {
    pub fn new() -> Self {
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sin", &["sin_kernel"]).unwrap();
        let f = dev.get_func("sin", "sin_kernel").unwrap();
        Self(dev, f)
    }
}

impl Operator for CudaSin {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        unsafe {
            self.1.clone().launch(
                LaunchConfig::for_num_elems(inp_size as u32),
                (&mut out, inp, inp_size as i32),
            )
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaSqrt(Arc<CudaDevice>, CudaFunction);
impl PartialEq for CudaSqrt {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSqrt {
    pub fn new() -> Self {
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void sqrt_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sqrt(inp[i]);
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sqrt", &["sqrt_kernel"]).unwrap();
        let f = dev.get_func("sqrt", "sqrt_kernel").unwrap();
        Self(dev, f)
    }
}

impl Operator for CudaSqrt {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        unsafe {
            self.1.clone().launch(
                LaunchConfig::for_num_elems(inp_size as u32),
                (&mut out, inp, inp_size as i32),
            )
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaRecip(Arc<CudaDevice>, CudaFunction);
impl PartialEq for CudaRecip {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaRecip {
    pub fn new() -> Self {
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void recip_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = __frcp_rn(inp[i]);
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "recip", &["recip_kernel"]).unwrap();
        let f = dev.get_func("recip", "recip_kernel").unwrap();
        Self(dev, f)
    }
}

impl Operator for CudaRecip {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size = tensors[0].1.n_physical_elements();
        let mut out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        unsafe {
            self.1.clone().launch(
                LaunchConfig::for_num_elems(inp_size as u32),
                (&mut out, inp, inp_size as i32),
            )
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

// Binary Ops

#[derive(Debug, Clone)]
pub struct CudaAdd(Arc<CudaDevice>, CudaFunction, ShapeTracker, ShapeTracker);
impl PartialEq for CudaAdd {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaAdd {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void add_kernel(float *out, const float *inp_a, const float *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = 
            ({a_valid_exp} != 0 ? inp_a[{a_idx_exp}] : 0.0) 
            + ({b_valid_exp} != 0 ? inp_b[{b_idx_exp}] : 0.0);
    }}
}}",
            a_shape
                .dims
                .into_iter()
                .chain(b_shape.dims.into_iter())
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "add", &["add_kernel"]).unwrap();
        let f = dev.get_func("add", "add_kernel").unwrap();
        Self(dev, f, a_shape, b_shape)
    }
}

impl Operator for CudaAdd {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            (inp_size as i32).as_kernel_param(),
        ];
        let mut added_dims = HashSet::new();
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[1].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        unsafe {
            self.1
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaMul(Arc<CudaDevice>, CudaFunction, ShapeTracker, ShapeTracker);
impl PartialEq for CudaMul {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMul {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mul_kernel(float *out, const float *inp_a, const float *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = 
            ({a_valid_exp} != 0 ? inp_a[{a_idx_exp}] : 0.0) 
            * ({b_valid_exp} != 0 ? inp_b[{b_idx_exp}] : 0.0);
    }}
}}",
            a_shape
                .dims
                .into_iter()
                .chain(b_shape.dims.into_iter())
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "mul", &["mul_kernel"]).unwrap();
        let f = dev.get_func("mul", "mul_kernel").unwrap();
        Self(dev, f, a_shape, b_shape)
    }
}

impl Operator for CudaMul {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            (inp_size as i32).as_kernel_param(),
        ];
        let mut added_dims = HashSet::new();
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[1].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        unsafe {
            self.1
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaMod(Arc<CudaDevice>, CudaFunction, ShapeTracker, ShapeTracker);
impl PartialEq for CudaMod {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMod {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mod_kernel(float *out, const float *inp_a, const float *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = fmod(
            {a_valid_exp} == 0 ? 0 : inp_a[{a_idx_exp}], 
            {b_valid_exp} == 0 ? 0 : inp_b[{b_idx_exp}]
        );
    }}
}}",
            a_shape
                .dims
                .into_iter()
                .chain(b_shape.dims.into_iter())
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "mod", &["mod_kernel"]).unwrap();
        let f = dev.get_func("mod", "mod_kernel").unwrap();
        Self(dev, f, a_shape, b_shape)
    }
}

impl Operator for CudaMod {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            (inp_size as i32).as_kernel_param(),
        ];
        let mut added_dims = HashSet::new();
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[1].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        unsafe {
            self.1
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaLessThan(Arc<CudaDevice>, CudaFunction, ShapeTracker, ShapeTracker);
impl PartialEq for CudaLessThan {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaLessThan {
    pub fn new(a_shape: ShapeTracker, b_shape: ShapeTracker) -> Self {
        let (a_idx_exp, a_valid_exp) = (a_shape.index_expression(), a_shape.valid_expression());
        let (b_idx_exp, b_valid_exp) = (b_shape.index_expression(), b_shape.valid_expression());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void lessthan_kernel(float *out, const float *inp_a, const float *inp_b, int numel{}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = inp_a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = inp_b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = 1.0;
        }} else {{
            out[idx] = 0.0;
        }}
    }}
}}",
            a_shape
                .dims
                .into_iter()
                .chain(b_shape.dims.into_iter())
                .filter_map(|d| if let Dim::Unknown(c) = d {
                    Some(c)
                } else {
                    None
                })
                .unique()
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "lessthan", &["lessthan_kernel"]).unwrap();
        let f = dev.get_func("lessthan", "lessthan_kernel").unwrap();
        Self(dev, f, a_shape, b_shape)
    }
}

impl Operator for CudaLessThan {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.n_elements();

        let out = unsafe { self.0.alloc::<f32>(inp_size) }.unwrap();
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            (inp_size as i32).as_kernel_param(),
        ];
        let mut added_dims = HashSet::new();
        for (d1, d2) in self.2.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[1].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added_dims.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added_dims.insert(c);
                }
            }
        }
        unsafe {
            self.1
                .clone()
                .launch_async_impl(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
        }
        .unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaSumReduce(usize, Arc<CudaDevice>, CudaFunction, ShapeTracker);

impl PartialEq for CudaSumReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaSumReduce {
    fn new(dim: usize, shape: ShapeTracker) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let ptx = compile_ptx(format!("
extern \"C\" __global__ void sumreduce_kernel(float *out, const float *inp, const int front_size, const int back_size, const int dim_size, int numel{}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if ({valid_exp} != 0) {{
                int a_idx = {idx_exp};
                reduce_value += inp[a_idx];
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sumreduce", &["sumreduce_kernel"])
            .unwrap();
        let f = dev.get_func("sumreduce", "sumreduce_kernel").unwrap();
        Self(dim, dev, f, shape)
    }
}
impl Operator for CudaSumReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.0);
        let inp_size = shape.n_elements();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.0)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.0 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.0].to_usize().unwrap();

        let out = unsafe { self.1.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut added = HashSet::new();
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added.insert(c);
                }
            }
        }
        unsafe { self.2.clone().launch_async_impl(cfg, &mut params) }.unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct CudaMaxReduce(usize, Arc<CudaDevice>, CudaFunction, ShapeTracker);

impl PartialEq for CudaMaxReduce {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMaxReduce {
    fn new(dim: usize, shape: ShapeTracker) -> Self {
        let idx_exp = shape.index_expression();
        let valid_exp = shape.valid_expression();
        let ptx = compile_ptx(format!("
extern \"C\" __global__ void maxreduce_kernel(float *out, const float *inp, const int front_size, const int back_size, const int dim_size, int numel{}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = -__int_as_float(0x7f800000);
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if ({valid_exp} != 0) {{
                int a_idx = {idx_exp};
                reduce_value = max(reduce_value, inp[a_idx]);
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "maxreduce", &["maxreduce_kernel"])
            .unwrap();
        let f = dev.get_func("maxreduce", "maxreduce_kernel").unwrap();
        Self(dim, dev, f, shape)
    }
}
impl Operator for CudaMaxReduce {
    fn process(&self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.0);
        let inp_size = shape.n_elements();
        let inp = tensors[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.0)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.0 + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.0].to_usize().unwrap();

        let out = unsafe { self.1.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut added = HashSet::new();
        for (d1, d2) in self.3.shape().into_iter().zip(tensors[0].1.shape()) {
            if let Dim::Unknown(c) = d1 {
                if !added.contains(&c) {
                    params.push(d2.to_usize().unwrap().as_kernel_param());
                    added.insert(c);
                }
            }
        }
        unsafe { self.2.clone().launch_async_impl(cfg, &mut params) }.unwrap();

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct CudaPrimitiveOptimizer;

impl GraphOptimizer for CudaPrimitiveOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // Go through the graph and insert copy ops
        // Copy to device
        for input_node in graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .node_weight(*n)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice)
                .input(input_node, 0, ShapeTracker::new(&[]))
                .finish();

            // Switch outgoing edges from input to copy_node
            for (edge_id, weight, dest) in graph
                .graph
                .edges_directed(input_node, petgraph::Direction::Outgoing)
                .map(|e| (e.id(), *e.weight(), e.target()))
                .filter(|(_, _, trg)| *trg != copy_node)
                .collect::<Vec<_>>()
            {
                graph.graph.add_edge(copy_node, dest, weight);
                graph.graph.remove_edge(edge_id);
            }

            if graph.to_retrieve.contains(&input_node) {
                graph.to_retrieve.insert(copy_node);
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
                        .next()
                        .unwrap()
                        .weight()
                        .2,
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyFromDevice)
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
                        .next()
                        .unwrap()
                        .id(),
                )
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let (source, shape) = (
                graph.graph.edge_endpoints(edge).unwrap().0,
                graph.graph.edge_weight(edge).unwrap().2,
            );
            let copy_node = graph
                .add_op(CudaCopyFromDevice)
                .input(source, 0, shape)
                .finish();
            graph.graph.add_edge(copy_node, output_node, (0, 0, shape));
            graph.graph.remove_edge(edge);
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .sorted_by_key(|e| e.weight().0)
                .map(|e| e.weight().2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::new());
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::new());
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::new());
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::new());
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::new());
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::new(src_shapes[0], src_shapes[1]));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::new(src_shapes[0], src_shapes[1]));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::new(src_shapes[0], src_shapes[1]));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::new(src_shapes[0], src_shapes[1]));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::new(src_shapes[0]));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref::<SumReduce>() {
                *op_ref = Box::new(CudaSumReduce::new(*dim, src_shapes[0]));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref::<MaxReduce>() {
                *op_ref = Box::new(CudaMaxReduce::new(*dim, src_shapes[0]));
            }
        }
    }
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}
