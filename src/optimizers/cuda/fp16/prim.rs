use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashSet},
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
use petgraph::visit::EdgeRef;

use crate::{op::*, prelude::*};

/// Copy a tensor to the GPU
#[derive(Debug, Clone)]
pub struct CudaCopyToDevice(Arc<CudaDevice>);
impl PartialEq for CudaCopyToDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaCopyToDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<CudaSlice<f16>>()
            || inp[0].0.borrowed().data.as_any().is::<CudaSlice<usize>>()
        {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        if let Some(cpu_data) = inp[0].0.borrowed().data.as_any().downcast_ref::<Vec<f32>>() {
            let vec = cpu_data
                .iter()
                .map(|i| f16::from_f32(*i))
                .collect::<Vec<_>>();
            let mut a = unsafe { self.0.alloc::<f16>(vec.len()).unwrap() };
            self.0.htod_copy_into(vec, &mut a).unwrap();
            vec![Tensor { data: Box::new(a) }]
        } else {
            let cpu_data = inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<usize>>()
                .unwrap();
            let mut a = unsafe { self.0.alloc::<usize>(cpu_data.len()).unwrap() };
            self.0.htod_sync_copy_into(cpu_data, &mut a).unwrap();
            vec![Tensor { data: Box::new(a) }]
        }
    }
}

/// Copy a tensor from the GPU
#[derive(Debug, Clone)]
pub struct CudaCopyFromDevice(Arc<CudaDevice>);
impl PartialEq for CudaCopyFromDevice {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl Operator for CudaCopyFromDevice {
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>()
            || inp[0].0.borrowed().data.as_any().is::<Vec<usize>>()
        {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        if let Some(cuda_data) = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
        {
            vec![Tensor {
                data: Box::new(
                    self.0
                        .dtoh_sync_copy(cuda_data)
                        .unwrap()
                        .into_iter()
                        .map(|i| i.to_f32())
                        .collect::<Vec<_>>(),
                ),
            }]
        } else {
            let cuda_data = inp[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<CudaSlice<usize>>()
                .unwrap();
            vec![Tensor {
                data: Box::new(self.0.dtoh_sync_copy(cuda_data).unwrap()),
            }]
        }
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
        out[idx] = __hmul(({a_valid_exp}) == 0 ? __float2half(0.0) : inp_a[{a_idx_exp}], ({b_valid_exp}) == 0 ? __float2half(0.0) : inp_b[{b_idx_exp}]);
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
                int a_idx = {idx_exp};
                reduce_value = __hadd(reduce_value, inp[a_idx]);
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
                .map(|c| format!(", int {c}"))
                .collect::<String>()
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
pub struct CudaPrimitiveOptimizer;

impl GraphOptimizer for CudaPrimitiveOptimizer {
    fn optimize(&self, graph: &mut Graph) {
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
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice(dev.clone()))
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

            // If there are inputs to this function remap the function to the copy node
            if graph
                .graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .count()
                != 0
            {
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
                    .add_op(CudaCopyFromDevice(dev.clone()))
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
                .add_op(CudaCopyFromDevice(dev.clone()))
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
                .add_op(CudaCopyFromDevice(dev.clone()))
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
                *op_ref = Box::new(CudaLog2::new(dev.clone()));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::new(dev.clone()));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::new(dev.clone()));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::new(dev.clone()));
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
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref::<SumReduce>() {
                *op_ref = Box::new(CudaSumReduce::new(*dim, src_shapes[0], dev.clone()));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref::<MaxReduce>() {
                *op_ref = Box::new(CudaMaxReduce::new(*dim, src_shapes[0], dev.clone()));
            }
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
