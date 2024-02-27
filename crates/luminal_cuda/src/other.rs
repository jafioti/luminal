use std::{marker::PhantomData, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
    shape::symbolic::BigExpression,
};
use rustc_hash::FxHashMap;

use crate::{
    binary::CudaSub,
    prim::{CudaAdd, CudaContiguous, CudaSumReduce},
    select_const, CudaData, CudaFloat,
};

#[derive(LuminalPrint, Clone, LuminalEqFalse)]
pub struct CudaARange<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    pub size: BigExpression,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}

impl<T: CudaFloat> CudaARange<T> {
    pub fn new(
        dev: Arc<CudaDevice>,
        size: BigExpression,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void arange({type_name} *out, int n_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"
        );
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
            "arange",
            &["arange"],
        )
        .unwrap();
        Self {
            function: dev.get_func("arange", "arange").unwrap(),
            device: dev,
            size,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T> Operator for CudaARange<T>
where
    T: std::fmt::Debug + Copy + cudarc::driver::DeviceRepr + std::marker::Unpin + CudaFloat,
    CudaData<T>: Data,
{
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let n_elements = self
            .size
            .exec(unsafe { self.dyn_map.as_ref().unwrap() })
            .unwrap();
        let mut out = self.device.alloc_zeros::<T>(n_elements).unwrap();
        unsafe {
            self.function
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(n_elements as u32),
                    (&mut out, n_elements as i32),
                )
                .unwrap();
        }

        vec![Tensor {
            data: Box::new(CudaData(out)),
        }]
    }
}

#[derive(LuminalPrint, Default)]
pub struct ARangeCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for ARangeCompiler<T>
where
    CudaData<T>: Data,
{
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        let (
            mut one_const,
            mut contig1,
            mut contig2,
            mut contig3,
            mut contig4,
            mut sum_reduce,
            mut subtraction_constant,
            mut subtraction,
        ) = (
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
            NodeIndex::default(),
        );

        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig = SelectOp::new().ty::<CudaContiguous<T>>();
        let pre_sub_pattern = select_const!(1.0, T)
            .ptr(&mut one_const)
            .edge(contig.clone().ptr(&mut contig1))
            .edge(contig.clone().ptr(&mut contig2))
            .edge(contig.clone().ptr(&mut contig3))
            .edge(contig.clone().ptr(&mut contig4))
            .edge(
                SelectOp::new()
                    .ty::<CudaSumReduce<T>>()
                    .ptr(&mut sum_reduce),
            );
        let mut s1 = pre_sub_pattern
            .clone()
            .edge(
                select_const!(1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<CudaSub<T>>().ptr(&mut subtraction)),
            )
            .search(graph);
        let mut s2 = pre_sub_pattern
            .edge(
                select_const!(-1.0, T)
                    .ptr(&mut subtraction_constant)
                    .edge(SelectOp::new().ty::<CudaAdd<T>>().ptr(&mut subtraction)),
            )
            .search(graph);

        while s1.next_match() || s2.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(one_const, contig1)
                            .next()
                            .unwrap()
                            .id(),
                    )
                    .unwrap()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(CudaARange::<T>::new(
                    dev.clone(),
                    arange_amount.into(),
                    &graph.dyn_map,
                ))
                .finish();
            move_outgoing_edge(subtraction, arange_op, &mut graph.graph);

            graph.graph.remove_node(subtraction);
            graph.safe_remove_node(subtraction_constant, 0);
            graph.safe_remove_node(sum_reduce, 0);
            graph.safe_remove_node(contig4, 0);
            graph.safe_remove_node(contig3, 0);
            graph.safe_remove_node(contig2, 0);
            graph.safe_remove_node(contig1, 0);
            graph.safe_remove_node(one_const, 0);
            s1.clear_cached_results();
            s2.clear_cached_results();
        }
    }
}
