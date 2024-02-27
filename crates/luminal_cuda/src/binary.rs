use std::{marker::PhantomData, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, DeviceRepr, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use itertools::Itertools;
use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
};
use rustc_hash::FxHashMap;

use crate::{
    get_idx_valid_exps, hash,
    other::CudaARange,
    prim::{CudaAdd, CudaCopyToDevice, CudaLessThan, CudaMul, CudaSumReduce},
    render_dyn_dim_inputs, select_const, CudaData, CudaFloat,
};

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct CudaSub<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}

impl<T: CudaFloat> CudaSub<T> {
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
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] =
            (({a_valid}) == 0 ? {} : inp_a[{a_idx}])
            - (({b_valid}) == 0 ? {} : inp_b[{b_idx}]);
    }}
}}",
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
        Self {
            function: dev.get_func(&name, &name).unwrap(),
            device: dev,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T> Operator for CudaSub<T>
where
    T: std::fmt::Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
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

        let out = self.device.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.dyn_map.as_ref().unwrap() };
        for (i, d) in self.dyn_symbols.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe {
                dims[0]
                    .as_kernel_param()
                    .add(i * std::mem::size_of::<i32>())
            });
        }
        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct CudaSubtractionCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CudaSubtractionCompiler<T>
where
    CudaData<T>: luminal::prelude::Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        let (mut neg_one, mut mul, mut add) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let mut searcher = select_const!(-1.0, T)
            .ptr(&mut neg_one)
            .edge(SelectOp::new().ty::<CudaMul<T>>().ptr(&mut mul))
            .edge(SelectOp::new().ty::<CudaAdd<T>>().ptr(&mut add))
            .search(graph);

        while searcher.next_match() {
            if check_no_delete(graph, &[neg_one, mul, add]) {
                continue;
            }
            let (a, a_edge) = graph
                .graph
                .edges_directed(add, petgraph::Direction::Incoming)
                .find(|e| e.source() != mul)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (b, b_edge) = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != neg_one)
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let b_final_shape = graph
                .graph
                .edges_connecting(mul, add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            if !b_final_shape.is_contiguous()
                || b_final_shape.is_sliced()
                || b_final_shape.is_padded()
            {
                continue;
            }
            let sub = graph
                .add_op(CudaSub::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    &graph.dyn_map,
                ))
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(add, sub, &mut graph.graph);

            if graph.get_dests(neg_one).len() == 1 {
                graph.graph.remove_node(neg_one);
            }
            graph.graph.remove_node(mul);
            graph.graph.remove_node(add);
        }
    }
}

#[derive(LuminalEqTrue, LuminalPrint, Clone)]
pub struct CudaEqual<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}

impl<T: CudaFloat> CudaEqual<T> {
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
        let mut code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        {type_name} a_val = ({a_valid}) == 0 ? {} : inp_a[{a_idx}];
        {type_name} b_val = ({b_valid}) == 0 ? {} : inp_b[{b_idx}];
        out[idx] = ({type_name})(a_val == b_val);
    }}
}}",
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
        Self {
            function: dev.get_func(&name, &name).unwrap(),
            device: dev,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T> Operator for CudaEqual<T>
where
    T: std::fmt::Debug
        + Copy
        + cudarc::driver::DeviceRepr
        + std::marker::Unpin
        + cudarc::driver::ValidAsZeroBits,
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

        let out = self.device.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            (&a.0).as_kernel_param(),
            (&b.0).as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        let mut dims = [0; 10];
        let dyn_map = unsafe { self.dyn_map.as_ref().unwrap() };
        for (i, d) in self.dyn_symbols.iter().enumerate() {
            dims[i] = dyn_map[d] as i32;
            params.push(unsafe {
                dims[0]
                    .as_kernel_param()
                    .add(i * std::mem::size_of::<i32>())
            });
        }
        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct CudaEqualCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CudaEqualCompiler<T>
where
    CudaData<T>: luminal::prelude::Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        let (mut less_than1, mut less_than2, mut add, mut one, mut sub) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = select_const!(1.0, T).ptr(&mut one).edge(
            SelectOp::new()
                .ty::<CudaLessThan<T>>()
                .ptr(&mut less_than1)
                .edge(
                    SelectOp::new()
                        .ty::<CudaLessThan<T>>()
                        .ptr(&mut less_than2)
                        .edge(SelectOp::new().ty::<CudaAdd<T>>().ptr(&mut add)),
                )
                .edge(SelectOp::new().ty::<CudaSub<T>>().ptr(&mut sub)),
        );

        let mut searcher = s.search(graph);
        while searcher.next_match() {
            let lt1_inputs = graph
                .graph
                .neighbors_directed(less_than1, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            let lt2_inputs = graph
                .graph
                .neighbors_directed(less_than2, petgraph::Direction::Incoming)
                .sorted()
                .collect::<Vec<_>>();
            if lt1_inputs != lt2_inputs {
                continue;
            }
            let inputs = graph
                .graph
                .edges_directed(less_than1, petgraph::Direction::Incoming)
                .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                .map(|e| e.source())
                .collect::<Vec<_>>();
            let (a, b) = (inputs[0], inputs[1]);
            if check_no_delete(graph, &[less_than1, less_than2, add, one, sub]) {
                continue;
            }
            let a_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(a, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let b_edge = graph
                .graph
                .edge_weight(
                    graph
                        .graph
                        .edges_connecting(b, less_than1)
                        .next()
                        .unwrap()
                        .id(),
                )
                .unwrap()
                .as_data()
                .unwrap();
            let equals = graph
                .add_op(CudaEqual::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    &graph.dyn_map,
                ))
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(sub, equals, &mut graph.graph);

            graph.graph.remove_node(sub);
            graph.safe_remove_node(add, 0);
            graph.safe_remove_node(one, 0);
            graph.safe_remove_node(less_than2, 0);
            graph.safe_remove_node(less_than1, 0);
            searcher.clear_cached_results();
        }
    }
}

#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct CudaGather<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    pub embed_dim: usize,
    _phantom: PhantomData<T>,
}

impl<T: CudaFloat> CudaGather<T> {
    pub fn new(dev: Arc<CudaDevice>, embed_dim: usize) -> Self {
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void gather({type_name} *out, const {type_name} *weights, const float *inp, int n_embeddings, int embedding_dim) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n_embeddings && y < embedding_dim) {{
        out[x * embedding_dim + y] = weights[(int)inp[x] * embedding_dim + y];
    }}
}}");
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
            "gather",
            &["gather"],
        )
        .unwrap();
        Self {
            function: dev.get_func("gather", "gather").unwrap(),
            device: dev,
            embed_dim,
            _phantom: Default::default(),
        }
    }
}

impl<T> Operator for CudaGather<T>
where
    T: std::fmt::Debug + Copy + cudarc::driver::DeviceRepr + std::marker::Unpin + CudaFloat,
    CudaData<T>: Data,
{
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Inp 1 should be Vec<f32> and inp 2 should be a CudaSlice<T>
        let indexes = inputs[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let weights = inputs[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaData<T>>()
            .unwrap();

        let mut indexes_buffer = unsafe { self.device.alloc::<f32>(indexes.len()).unwrap() };
        self.device
            .htod_copy_into(indexes.clone(), &mut indexes_buffer)
            .unwrap();
        let mut out = self
            .device
            .alloc_zeros::<T>(indexes.len() * self.embed_dim)
            .unwrap();
        unsafe {
            self.function
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (
                            indexes.len().div_ceil(16) as u32,
                            self.embed_dim.div_ceil(16) as u32,
                            1,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut out,
                        &weights.0,
                        &indexes_buffer,
                        indexes.len(),
                        self.embed_dim,
                    ),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct MetalGatherCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for MetalGatherCompiler<T>
where
    CudaData<T>: luminal::prelude::Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        let (mut ind_copy, mut arange, mut equal, mut mul, mut sum_reduce) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );
        let s = SelectOp::new()
            .ty::<CudaARange<T>>()
            .ptr(&mut arange)
            .edge(
                SelectOp::new()
                    .ty::<CudaCopyToDevice<T>>()
                    .ptr(&mut ind_copy)
                    .edge(SelectOp::new().ty::<CudaEqual<T>>().ptr(&mut equal)),
            )
            .edge(SelectOp::new().ty::<CudaMul<T>>().ptr(&mut mul))
            .edge(
                SelectOp::new()
                    .ty::<CudaSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            );
        let mut searcher = s.search(graph);
        while searcher.next_match() {
            if check_no_delete(graph, &[arange, equal, mul, sum_reduce]) {
                continue;
            }
            let embedding_dim = graph
                .graph
                .edges_directed(mul, petgraph::Direction::Incoming)
                .find(|e| e.source() != equal && !e.weight().is_schedule())
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2
                .shape()[2]
                .to_usize()
                .unwrap();
            let gather = graph
                .add_op(CudaGather::<T>::new(dev.clone(), embedding_dim))
                .finish();
            move_incoming_edge(ind_copy, gather, &mut graph.graph);
            graph.safe_remove_node(equal, 1);
            move_incoming_edge(mul, gather, &mut graph.graph);
            move_outgoing_edge(sum_reduce, gather, &mut graph.graph);
            graph.graph.remove_node(sum_reduce);
            graph.safe_remove_node(mul, 0);
            graph.safe_remove_node(ind_copy, 0);
            graph.safe_remove_node(arange, 0);
        }
    }
}
