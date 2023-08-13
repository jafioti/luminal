use std::any::{Any, TypeId};

use cudarc::{
    driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use petgraph::{stable_graph::NodeIndex, visit::EdgeRef};

use crate::{op::*, prelude::*};

/// Copy a tensor to the GPU
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCopyToDevice;

impl Operator for CudaCopyToDevice {
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let dev = CudaDevice::new(0).unwrap();
        let cpu_data = inp[0].0.data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut a: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
        dev.htod_sync_copy_into(cpu_data, &mut a).unwrap();
        (
            Some(Tensor { data: Box::new(a) }),
            TensorView {
                tensor_id: i,
                shape: inp[0].1.shape.clone(),
            },
        )
    }
}

/// Copy a tensor from the GPU
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCopyFromDevice;

impl Operator for CudaCopyFromDevice {
    fn process(
        &self,
        inp: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let dev = CudaDevice::new(0).unwrap();
        let cuda_data = inp[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let a = dev.dtoh_sync_copy(cuda_data).unwrap();
        (
            Some(Tensor { data: Box::new(a) }),
            TensorView {
                tensor_id: i,
                shape: inp[0].1.shape.clone(),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaContiguous;
impl Operator for CudaContiguous {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let view = &tensors[0].1;
        if view.shape.is_contiguous() {
            // It's already contiguous
            return (None, view.clone());
        }
        let res_shape = view.shape.shape().clone();
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (idx, valid) = tensors[0].1.shape.index_node();
        let valid_exp = valid.to_string_no_range();
        let idx_exp = idx.to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void contiguous_kernel(float *out, const float *a, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {idx_exp};
    if (idx < numel && {valid_exp} != 0) {{
        out[idx] = a[a_idx];
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "add", &["add_kernel"]).unwrap();
        let f = dev.get_func("add", "add_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

// Unary Op (A -> A)

#[derive(Debug, Clone, PartialEq)]
pub struct CudaLog2;
impl Operator for CudaLog2 {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.shape.shape().iter().product();
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

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, inp, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: tensors[0].1.shape.clone(),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaExp2;
impl Operator for CudaExp2 {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.shape.shape().iter().product();
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

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, inp, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: tensors[0].1.shape.clone(),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaSin;
impl Operator for CudaSin {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.shape.shape().iter().product();
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

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, inp, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: tensors[0].1.shape.clone(),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaSqrt;
impl Operator for CudaSqrt {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.shape.shape().iter().product();
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

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, inp, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: tensors[0].1.shape.clone(),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaRecip;
impl Operator for CudaRecip {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].1.shape.shape().iter().product();
        let ptx = compile_ptx(
            "
extern \"C\" __global__ void recip_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = 1.0 / inp[i];
    }
}",
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "recip", &["recip_kernel"]).unwrap();
        let f = dev.get_func("recip", "recip_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, inp, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: tensors[0].1.shape.clone(),
            },
        )
    }
}

// Binary Ops

#[derive(Debug, Clone, PartialEq)]
pub struct CudaAdd;
impl Operator for CudaAdd {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = tensors[0].1.shape.get_real_shape([&tensors[1].1.shape]);
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());
        let (b_idx, b_valid) = tensors[1].1.shape.index_node();
        let (b_idx_exp, b_valid_exp) = (b_idx.to_string_no_range(), b_valid.to_string_no_range());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void add_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = b[{b_idx_exp}];
        }}
        out[idx] = a_t + b_t;
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "add", &["add_kernel"]).unwrap();
        let f = dev.get_func("add", "add_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaMul;
impl Operator for CudaMul {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = tensors[0].1.shape.get_real_shape([&tensors[1].1.shape]);
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());
        let (b_idx, b_valid) = tensors[1].1.shape.index_node();
        let (b_idx_exp, b_valid_exp) = (b_idx.to_string_no_range(), b_valid.to_string_no_range());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mul_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = b[{b_idx_exp}];
        }}
        out[idx] = a_t * b_t;
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "mul", &["mul_kernel"]).unwrap();
        let f = dev.get_func("mul", "mul_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaMax;
impl Operator for CudaMax {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = tensors[0].1.shape.get_real_shape([&tensors[1].1.shape]);
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());
        let (b_idx, b_valid) = tensors[1].1.shape.index_node();
        let (b_idx_exp, b_valid_exp) = (b_idx.to_string_no_range(), b_valid.to_string_no_range());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void max_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = b[{b_idx_exp}];
        }}
        out[idx] = max(a_t, b_t);
    }}
}}"
        ))
        .unwrap();
        let dev: std::sync::Arc<CudaDevice> = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "max", &["max_kernel"]).unwrap();
        let f = dev.get_func("max", "max_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaMod;
impl Operator for CudaMod {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = tensors[0].1.shape.get_real_shape([&tensors[1].1.shape]);
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());
        let (b_idx, b_valid) = tensors[1].1.shape.index_node();
        let (b_idx_exp, b_valid_exp) = (b_idx.to_string_no_range(), b_valid.to_string_no_range());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mod_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = b[{b_idx_exp}];
        }}
        out[idx] = fmod(a_t, b_t);
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "mod", &["mod_kernel"]).unwrap();
        let f = dev.get_func("mod", "mod_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaLessThan;
impl Operator for CudaLessThan {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let res_shape = tensors[0].1.shape.get_real_shape([&tensors[1].1.shape]);
        let a = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());
        let (b_idx, b_valid) = tensors[1].1.shape.index_node();
        let (b_idx_exp, b_valid_exp) = (b_idx.to_string_no_range(), b_valid.to_string_no_range());
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void less_than_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        float a_t = 0.0;
        float b_t = 0.0;
        if ({a_valid_exp} != 0) {{
            a_t = a[{a_idx_exp}];
        }}
        if ({b_valid_exp} != 0) {{
            b_t = b[{b_idx_exp}];
        }}
        if (a_t < b_t) {{
            out[idx] = 1.0;
        }} else {{
            out[idx] = 0.0;
        }}
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "less_than", &["less_than_kernel"])
            .unwrap();
        let f = dev.get_func("less_than", "less_than_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: ShapeTracker::new(res_shape),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaSumReduce(pub usize);
impl Operator for CudaSumReduce {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0].1.shape.shape().iter().take(self.0).product();
        let back_size: usize = tensors[0].1.shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = tensors[0].1.shape.shape()[self.0];
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());

        let ptx = compile_ptx(
            format!("
extern \"C\" __global__ void sumreduce_kernel(float *out, const float *inp, const int front_size, const int back_size, const int dim_size, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numel) {{
        int a = i / back_size;
        int b = i % back_size;
        float reduce_value = 0.0;
        for (int c = 0; c < dim_size; c++) {{
            int idx = a * dim_size * back_size + c * back_size + b;
            if ({a_valid_exp} != 0) {{
                int a_idx = {a_idx_exp};
                reduce_value += inp[a_idx];
            }}
        }}
        out[i] = reduce_value;
    }}
}}"),
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sumreduce", &["sumreduce_kernel"])
            .unwrap();
        let f = dev.get_func("sumreduce", "sumreduce_kernel").unwrap();

        let mut shape_tracker = tensors[0].1.shape.clone();
        let mut prev_shape = shape_tracker.shape().clone();
        prev_shape.remove(self.0);
        let result_size = prev_shape.iter().product();
        let mut out = dev.alloc_zeros::<f32>(result_size).unwrap();
        let cfg = LaunchConfig::for_num_elems(result_size as u32);
        unsafe {
            f.launch(
                cfg,
                (
                    &mut out,
                    inp,
                    front_size as i32,
                    back_size as i32,
                    dim_size as i32,
                    result_size as i32,
                ),
            )
        }
        .unwrap();

        shape_tracker.reshape(prev_shape);
        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: shape_tracker,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaMaxReduce(pub usize);
impl Operator for CudaMaxReduce {
    fn process(
        &self,
        tensors: Vec<(&Tensor, TensorView)>,
        i: NodeIndex,
    ) -> (Option<Tensor>, TensorView) {
        let inp = tensors[0]
            .0
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0].1.shape.shape().iter().take(self.0).product();
        let back_size: usize = tensors[0].1.shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = tensors[0].1.shape.shape()[self.0];
        let (a_idx, a_valid) = tensors[0].1.shape.index_node();
        let (a_idx_exp, a_valid_exp) = (a_idx.to_string_no_range(), a_valid.to_string_no_range());

        let ptx = compile_ptx(
            format!("
extern \"C\" __global__ void maxreduce_kernel(float *out, const float *inp, const int front_size, const int back_size, const int dim_size, int numel) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numel) {{
        int a = i / back_size;
        int b = i % back_size;
        float reduce_value = -__int_as_float(0x7f800000);
        for (int c = 0; c < dim_size; c++) {{
            int idx = a * dim_size * back_size + c * back_size + b;
            if ({a_valid_exp} != 0) {{
                int a_idx = {a_idx_exp};
                reduce_value = max(reduce_value, inp[a_idx]);
            }}
        }}
        out[i] = reduce_value;
    }}
}}"),
        )
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "maxreduce", &["maxreduce_kernel"])
            .unwrap();
        let f = dev.get_func("maxreduce", "maxreduce_kernel").unwrap();

        let mut shape_tracker = tensors[0].1.shape.clone();
        let mut prev_shape = shape_tracker.shape().clone();
        prev_shape.remove(self.0);
        let result_size = prev_shape.iter().product();
        let mut out = dev.alloc_zeros::<f32>(result_size).unwrap();
        let cfg = LaunchConfig::for_num_elems(result_size as u32);
        unsafe {
            f.launch(
                cfg,
                (
                    &mut out,
                    inp,
                    front_size as i32,
                    back_size as i32,
                    dim_size as i32,
                    result_size as i32,
                ),
            )
        }
        .unwrap();

        shape_tracker.reshape(prev_shape);
        (
            Some(Tensor {
                data: Box::new(out),
            }),
            TensorView {
                tensor_id: i,
                shape: shape_tracker,
            },
        )
    }
}

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct CudaPrimitiveOptimizer;

impl GraphOptimizer for CudaPrimitiveOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // Go through the graph and insert copy ops
        // Copy to device
        for (input_node, input_shape) in graph
            .graph
            .node_indices()
            .filter(|n| {
                graph
                    .graph
                    .node_weight(*n)
                    .unwrap()
                    .0
                    .as_any()
                    .is::<Function>()
            })
            .map(|n| (n, graph.graph.node_weight(n).unwrap().1.clone()))
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice, input_shape)
                .input(input_node)
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

        // Copy from device
        for (output_node, output_shape) in graph
            .to_retrieve
            .iter()
            // Filter non-functions
            .filter(|n| {
                !graph
                    .graph
                    .node_weight(**n)
                    .unwrap()
                    .0
                    .as_any()
                    .is::<Function>()
            })
            .map(|n| (*n, graph.graph.node_weight(*n).unwrap().1.clone()))
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyFromDevice, output_shape)
                .input(output_node)
                .finish();

            move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                output_node,
                copy_node,
            );
        }

        // Swap primitive ops
        for id in graph.graph.node_indices().collect::<Vec<_>>() {
            let op = graph.graph.node_weight(id).unwrap().0.as_any().type_id();
            let op_ref = &mut graph.graph.node_weight_mut(id).unwrap().0;
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2);
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2);
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin);
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt);
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip);
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd);
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul);
            } else if is::<Max>(op) {
                *op_ref = Box::new(CudaMax);
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod);
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan);
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous);
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref::<SumReduce>() {
                *op_ref = Box::new(CudaSumReduce(*dim));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref::<MaxReduce>() {
                *op_ref = Box::new(CudaMaxReduce(*dim));
            }
        }
    }
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}
