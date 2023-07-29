use std::any::Any;

use cudarc::{
    driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use itertools::Itertools;
use petgraph::visit::EdgeRef;

use crate::{
    op::{MaxReduce, Operator, SumReduce},
    prelude::*,
};

// Ops and optimizers specific to CUDA execution

pub type CudaOptimizer = (CudaPrimitiveOptimizer,);

impl Data for CudaSlice<f32> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
            .filter(|n| graph.graph.node_weight(*n).unwrap().0.name() == "Function")
            .map(|n| (n, graph.graph.node_weight(n).unwrap().1.clone()))
            .collect_vec()
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
                .collect_vec()
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
            .filter(|n| graph.graph.node_weight(**n).unwrap().0.name() != "Function")
            .map(|n| (*n, graph.graph.node_weight(*n).unwrap().1.clone()))
            .collect_vec()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyFromDevice, output_shape)
                .input(output_node)
                .finish();

            Graph::move_references(
                &mut graph.id_remap,
                &mut graph.no_delete,
                &mut graph.to_retrieve,
                output_node,
                copy_node,
            );
        }

        // Swap primitive ops
        for (id, name) in graph
            .graph
            .node_indices()
            .map(|n| (n, graph.graph.node_weight(n).unwrap().0.name()))
            .collect_vec()
        {
            match name {
                "Log2" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaLog2),
                "Exp2" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaExp2),
                "Sin" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaSin),
                "Sqrt" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaSqrt),
                "Recip" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaRecip),
                "Add" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaAdd),
                "Sub" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaSub),
                "Mul" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaMul),
                "Div" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaDiv),
                "Max" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaMax),
                "Mod" => graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaMod),
                "SumReduce" => {
                    let dim = graph
                        .graph
                        .node_weight(id)
                        .unwrap()
                        .0
                        .as_any()
                        .downcast_ref::<SumReduce>()
                        .unwrap()
                        .0;
                    graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaSumReduce(dim));
                }
                "MaxReduce" => {
                    let dim = graph
                        .graph
                        .node_weight(id)
                        .unwrap()
                        .0
                        .as_any()
                        .downcast_ref::<MaxReduce>()
                        .unwrap()
                        .0;
                    graph.graph.node_weight_mut(id).unwrap().0 = Box::new(CudaMaxReduce(dim));
                }
                _ => {}
            };
        }
    }
}

/// Copy a tensor to the GPU
#[derive(Debug)]
pub struct CudaCopyToDevice;

impl Operator for CudaCopyToDevice {
    fn name(&self) -> &'static str {
        "CudaCopyToDevice"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        let dev = CudaDevice::new(0).unwrap();
        let cpu_data = inp[0].data.as_any().downcast_ref::<Vec<f32>>().unwrap();
        let mut a: CudaSlice<f32> = dev.alloc_zeros::<f32>(cpu_data.len()).unwrap();
        dev.htod_sync_copy_into(cpu_data, &mut a).unwrap();
        Tensor {
            data: Box::new(a),
            shape: inp[0].shape.clone(),
        }
    }
}

/// Copy a tensor from the GPU
#[derive(Debug)]
pub struct CudaCopyFromDevice;

impl Operator for CudaCopyFromDevice {
    fn name(&self) -> &'static str {
        "CudaCopyFromDevice"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, inp: Vec<&Tensor>) -> Tensor {
        let dev = CudaDevice::new(0).unwrap();
        let cuda_data = inp[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let a = dev.dtoh_sync_copy(cuda_data).unwrap();
        Tensor {
            data: Box::new(a),
            shape: inp[0].shape.clone(),
        }
    }
}

// Unary Op (A -> A)

#[derive(Debug, Clone)]
pub struct CudaLog2;
impl Operator for CudaLog2 {
    fn name(&self) -> &'static str {
        "CudaLog2"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].shape.shape().iter().product();
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

        Tensor {
            data: Box::new(out),
            shape: tensors[0].shape.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaExp2;
impl Operator for CudaExp2 {
    fn name(&self) -> &'static str {
        "CudaExp2"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].shape.shape().iter().product();
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

        Tensor {
            data: Box::new(out),
            shape: tensors[0].shape.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaSin;
impl Operator for CudaSin {
    fn name(&self) -> &'static str {
        "CudaSin"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].shape.shape().iter().product();
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

        Tensor {
            data: Box::new(out),
            shape: tensors[0].shape.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaSqrt;
impl Operator for CudaSqrt {
    fn name(&self) -> &'static str {
        "CudaSqrt"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].shape.shape().iter().product();
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

        Tensor {
            data: Box::new(out),
            shape: tensors[0].shape.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaRecip;
impl Operator for CudaRecip {
    fn name(&self) -> &'static str {
        "CudaRecip"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = tensors[0].shape.shape().iter().product();
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

        Tensor {
            data: Box::new(out),
            shape: tensors[0].shape.clone(),
        }
    }
}

// Binary Ops

#[derive(Debug, Clone)]
pub struct CudaAdd;
impl Operator for CudaAdd {
    fn name(&self) -> &'static str {
        "CudaAdd"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void add_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = a[a_idx] + b[b_idx];
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

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaSub;
impl Operator for CudaSub {
    fn name(&self) -> &'static str {
        "CudaSub"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void sub_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = a[a_idx] - b[b_idx];
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "sub", &["sub_kernel"]).unwrap();
        let f = dev.get_func("sub", "sub_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaMul;
impl Operator for CudaMul {
    fn name(&self) -> &'static str {
        "CudaMul"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mul_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = a[a_idx] * b[b_idx];
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

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaDiv;
impl Operator for CudaDiv {
    fn name(&self) -> &'static str {
        "CudaDiv"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void div_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = a[a_idx] / b[b_idx];
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "div", &["div_kernel"]).unwrap();
        let f = dev.get_func("div", "div_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaMax;
impl Operator for CudaMax {
    fn name(&self) -> &'static str {
        "CudaMax"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void max_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = max(a[a_idx], b[b_idx]);
    }}
}}"
        ))
        .unwrap();
        let dev = CudaDevice::new(0).unwrap();
        dev.load_ptx(ptx, "max", &["max_kernel"]).unwrap();
        let f = dev.get_func("max", "max_kernel").unwrap();

        let mut out = unsafe { dev.alloc::<f32>(inp_size) }.unwrap();
        let cfg = LaunchConfig::for_num_elems(inp_size as u32);
        unsafe { f.launch(cfg, (&mut out, a, b, inp_size as i32)) }.unwrap();

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaMod;
impl Operator for CudaMod {
    fn name(&self) -> &'static str {
        "CudaMod"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let res_shape = tensors[0]
            .shape
            .get_real_shape([&tensors[1].shape])
            .unwrap();
        let a = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let b = tensors[1]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let inp_size: usize = res_shape.iter().product();
        let a_index_fn_exp = tensors[0].shape.index_fn_node().to_string_no_range();
        let b_index_fn_exp = tensors[1].shape.index_fn_node().to_string_no_range();
        let ptx = compile_ptx(format!(
            "
extern \"C\" __global__ void mod_kernel(float *out, const float *a, const float *b, int numel) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = {a_index_fn_exp};
    int b_idx = {b_index_fn_exp};
    if (idx < numel) {{
        out[idx] = fmod(a[a_idx], b[b_idx]);
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

        Tensor {
            data: Box::new(out),
            shape: ShapeTracker::new(res_shape),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaSumReduce(pub usize);
impl Operator for CudaSumReduce {
    fn name(&self) -> &'static str {
        "CudaSumReduce"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0].shape.shape().iter().take(self.0).product();
        let back_size: usize = tensors[0].shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = tensors[0].shape.shape()[self.0];
        let inp_idx_exp = tensors[0].shape.index_fn_node().to_string_no_range();

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
            int a_idx = {inp_idx_exp};
            reduce_value += inp[a_idx];
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

        let mut shape_tracker = tensors[0].shape.clone();
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
        Tensor {
            data: Box::new(out),
            shape: shape_tracker,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaMaxReduce(pub usize);
impl Operator for CudaMaxReduce {
    fn name(&self) -> &'static str {
        "CudaMaxReduce"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn process(&self, tensors: Vec<&Tensor>) -> Tensor {
        let inp = tensors[0]
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f32>>()
            .unwrap();
        let front_size: usize = tensors[0].shape.shape().iter().take(self.0).product();
        let back_size: usize = tensors[0].shape.shape().iter().skip(self.0 + 1).product();
        let dim_size = tensors[0].shape.shape()[self.0];
        let inp_idx_exp = tensors[0].shape.index_fn_node().to_string_no_range();

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
            int a_idx = {inp_idx_exp};
            reduce_value = max(reduce_value, inp[a_idx]);
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

        let mut shape_tracker = tensors[0].shape.clone();
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
        Tensor {
            data: Box::new(out),
            shape: shape_tracker,
        }
    }
}

#[cfg(test)]
mod tests {
    use dfdx::prelude::{Module as DfdxModule, *};
    use itertools::Itertools;

    use super::CudaOptimizer;
    use crate::{
        nn::{activation::ReLU, linear::Linear},
        prelude::{Module, *},
        tests::{assert_close, assert_close_data},
    };

    #[test]
    fn test_log2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.log_2();
        b.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        assert_close_data(
            &b.retrieve().unwrap().real_data().unwrap(),
            &vec![1., 2., 3.]
                .into_iter()
                .map(|i: f32| i.log2())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_exp2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.exp_2();
        b.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        assert_close_data(
            &b.retrieve().unwrap().real_data().unwrap(),
            &vec![1., 2., 3.]
                .into_iter()
                .map(|i: f32| i.exp2())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_log2exp2() {
        // We can't use dfdx because it doesn't implement this op
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.exp_2().log_2();
        b.mark();

        cx.optimize(<(GenericOptimizer, CudaOptimizer)>::default());
        cx.execute();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &[1., 2., 3.]);
    }

    #[test]
    fn test_recip() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.recip();
        b.mark();
        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.recip();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.sin();
        b.mark();
        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sin();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }

    #[test]
    fn test_sqrt() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = a.sqrt();
        b.mark();
        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_a.sqrt();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }

    #[test]
    fn test_add() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a + b;
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a + d_b;

        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    }

    #[test]
    fn test_sub() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a - b;
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a - d_b;

        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    }

    #[test]
    fn test_mul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a * b;
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a * d_b;

        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    }

    #[test]
    fn test_div() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a / b;
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a / d_b;

        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    }

    #[test]
    fn test_max() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a.max(b);
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([1., 2., 3.]);
        let d_c = d_a.maximum(d_b);

        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
    }

    #[test]
    fn test_mod() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>();
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R1<3>>();
        b.set(vec![1., 2., 3.]);
        let c = a % b;
        c.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        // No dfdx equivalent

        assert_close_data(
            &c.retrieve().unwrap().real_data().unwrap(),
            &[1., 2., 3.]
                .into_iter()
                .zip([1., 2., 3.].into_iter())
                .map(|(a, b)| a % b)
                .collect_vec(),
        );
    }

    // Reduction op tests

    #[test]
    fn test_sum_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = a.sum_reduce::<_, crate::prelude::Axis<1>>();
        let c = a.sum_reduce::<_, crate::prelude::Axis<0>>();
        let d = a.sum_reduce::<_, crate::prelude::Axis<2>>();
        b.mark();
        c.mark();
        d.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().sum::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().sum::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.sum::<_, dfdx::shapes::Axis<2>>();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
        assert_close_data(&d.retrieve().unwrap().real_data().unwrap(), &d_d.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = a.max_reduce::<_, crate::prelude::Axis<1>>();
        let c = a.max_reduce::<_, crate::prelude::Axis<0>>();
        let d = a.max_reduce::<_, crate::prelude::Axis<2>>();
        b.mark();
        c.mark();
        d.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
        let d_b = d_a.clone().max::<_, dfdx::shapes::Axis<1>>();
        let d_c = d_a.clone().max::<_, dfdx::shapes::Axis<0>>();
        let d_d = d_a.max::<_, dfdx::shapes::Axis<2>>();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
        assert_close_data(&c.retrieve().unwrap().real_data().unwrap(), &d_c.as_vec());
        assert_close_data(&d.retrieve().unwrap().real_data().unwrap(), &d_d.as_vec());
    }

    #[test]
    fn test_mean_reduce() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>();
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = a.mean_reduce::<_, crate::prelude::Axis<1>>();
        b.mark();

        cx.optimize(CudaOptimizer::default());
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_a.mean::<_, dfdx::shapes::Axis<1>>();

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }

    #[test]
    fn test_relu_and_linear() {
        // Test single and batch, unoptimized and optimized
        let mut cx = Graph::new();
        let batch = cx.new_tensor::<R2<2, 3>>();
        let a = cx.new_tensor::<R1<3>>();

        let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&mut cx);
        model
            .0
            .weight
            .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
        let b = model.forward(a);
        let batch_out = model.forward(batch);

        a.set(vec![1.0, 2.0, 3.0]);
        batch.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        b.mark();
        batch_out.mark();
        cx.execute();

        let unoptimized_b = b.retrieve().unwrap();
        let unoptimized_batch_out = batch_out.retrieve().unwrap();

        cx.optimize(<(CudaOptimizer, GenericOptimizer)>::default());
        cx.execute();

        assert_close(&unoptimized_b, &b.retrieve().unwrap());
        assert_close(&unoptimized_batch_out, &batch_out.retrieve().unwrap());

        // Test against dfdx
        let dev = Cpu::default();
        let mut model = <(
            dfdx::nn::modules::builders::UnbiasedLinear<3, 4>,
            dfdx::nn::modules::builders::ReLU,
            dfdx::nn::modules::builders::UnbiasedLinear<4, 2>,
        )>::build_on_device(&dev);
        // Set weights
        model.0.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        model.2.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<2>),
            )
            .permute();
        let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
        let out = model.forward(a);

        assert_close_data(&unoptimized_b.real_data().unwrap(), &out.as_vec());
    }

    #[test]
    fn test_transformer_encoder_block() {
        let mut cx = Graph::new();
        let model: crate::nn::transformer::encoder::TransformerEncoderBlock<3, 4, 1> =
            InitModule::initialize(&mut cx);
        model
            .attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .ff
            .0
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model
            .ff
            .2
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

        let a = cx.new_tensor::<(usize, crate::shape::Const<3>)>();
        let b = model.forward(a);

        a.set_dyn(vec![-1., 2., 3., 3., 3., -1.], vec![2, 3]);
        b.mark();

        cx.optimize(<(CudaOptimizer, GenericOptimizer)>::default());
        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
            d_dev
                .build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>(
                );
        d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
        d_model.self_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.ff.0 .0.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        d_model.ff.0 .0.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
        d_model.ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.epsilon = 1e-5;
        d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.epsilon = 1e-5;
        let d_a = d_dev.tensor_from_vec(
            vec![-1., 2., 3., 3., 3., -1.],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward(d_a);

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
    }
}
