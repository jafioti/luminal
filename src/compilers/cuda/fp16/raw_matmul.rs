use std::sync::Arc;

use cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use half::f16;
use itertools::Itertools;
use petgraph::stable_graph::NodeIndex;

use crate::{
    op::{InputTensor, Operator},
    optimizers::cuda::hash,
    prelude::*,
};

use super::prim::{CudaMul, CudaSumReduce};

/// Multiplies a MxK matrix with a KxN matrix, resulting in a MxN matrix
#[derive(Debug, Clone)]
pub struct CudaMatmul2D(CudaFunction, Arc<CudaDevice>);
impl PartialEq for CudaMatmul2D {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl CudaMatmul2D {
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
        Self(dev.get_func(&name, &name).unwrap(), dev)
    }
}

impl Operator for CudaMatmul2D {
    fn process(&self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
        let (a_strides, b_strides) = (inp[0].1.strides(), inp[1].1.strides());
        let (m, k, n) = (
            a_shape[0].to_usize().unwrap() as i32,
            a_shape[1].to_usize().unwrap() as i32,
            b_shape[1].to_usize().unwrap() as i32,
        );
        let a = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let b = inp[1]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<f16>>()
            .unwrap();
        let mut out = self.1.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let (a_row_major, b_row_major) = (a_strides[0] > a_strides[1], b_strides[0] > b_strides[1]);
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };

        vec![Tensor {
            data: Box::new(out),
        }]
    }
}

#[derive(Default)]
pub struct CudaMatMulOptimizer;

impl GraphOptimizer for CudaMatMulOptimizer {
    fn optimize(&self, graph: &mut Graph) {
        // let dev = CudaDevice::new(0).unwrap();
        // // Look for the matmul pattern
        // let s = GraphSelector::default();
        // let (mut sum_reduce, mut mul) = (NodeIndex::default(), NodeIndex::default());
        // // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // // Actually starts at [A,B] | [B, C]
        // s.edge(
        //     s.op()
        //         .ty::<CudaMul>()
        //         .shapes(vec![
        //             vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
        //             vec![Dim::Unknown('A'), Dim::Unknown('C'), Dim::Unknown('B')],
        //         ])
        //         .fakes(vec![vec![false, true, false], vec![true, false, false]])
        //         .ptr(&mut mul),
        //     0,
        //     s.op()
        //         .ty::<CudaSumReduce>()
        //         .check(|o, _| {
        //             if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce>() {
        //                 o.2 == 2
        //             } else {
        //                 false
        //             }
        //         })
        //         .ptr(&mut sum_reduce),
        // );
        // for _ in s.search(graph) {
        //     if graph.no_delete.contains(&mul) {
        //         // The intermediate mul can't be deleted
        //         continue;
        //     }
        //     // Insert MatMul2D op
        //     let mut srcs = graph.get_sources(mul);
        //     // Undo expansions and permute
        //     srcs[0].1.remove_dim(1);
        //     srcs[1].1.remove_dim(0);
        //     srcs[1].1.permute(&[1, 0]);
        //     let new_op = graph
        //         .add_op(CudaMatmul2D(
        //             CudaBlas::new(dev.clone()).unwrap(),
        //             dev.clone(),
        //         ))
        //         .input(srcs[0].0, 0, srcs[0].1)
        //         .input(srcs[1].0, 0, srcs[1].1)
        //         .finish();

        //     // Create edges to dests
        //     move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
        //     move_references(
        //         &mut graph.id_remap,
        //         &mut graph.no_delete,
        //         &mut graph.to_retrieve,
        //         sum_reduce,
        //         new_op,
        //     );
        //     move_references(
        //         &mut graph.id_remap,
        //         &mut graph.no_delete,
        //         &mut graph.to_retrieve,
        //         mul,
        //         new_op,
        //     );

        //     // Remove the old ops
        //     graph.graph.remove_node(mul);
        //     graph.graph.remove_node(sum_reduce);
        // }
    }
}
