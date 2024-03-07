use std::{marker::PhantomData, sync::Arc};

use luminal_cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
    shape::symbolic::BigExpression,
};
use rustc_hash::FxHashMap;

use crate::{
    binary::CudaSub,
    compile_and_load_kernel, constant,
    prim::{CudaContiguous, CudaSumReduce},
    CudaData, CudaFloat,
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
        device: Arc<CudaDevice>,
        size: BigExpression,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, int n_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"
        );
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            size,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaARange<T> {
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

impl<T: CudaFloat> Compiler for ARangeCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let one = constant::<T>(1.);
        let contig1 = unary::<CudaContiguous<T>>(one.clone());
        let sum_reduce =
            unary::<CudaSumReduce<T>>(unary::<CudaContiguous<T>>(unary::<CudaContiguous<T>>(
                unary::<CudaContiguous<T>>(contig1.clone()),
            )));
        let sub = binary::<CudaSub<T>>(sum_reduce, one.clone());
        let mut s = sub.clone().search(graph);

        while s.next_match() {
            let arange_amount = {
                let sh = graph
                    .graph
                    .edge_weight(
                        graph
                            .graph
                            .edges_connecting(s.get(&one), s.get(&contig1))
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
            move_outgoing_edge(s.get(&sub), arange_op, &mut graph.graph);
            graph.graph.remove_node(s.get(&sub));
            s.try_delete();
        }
    }
}
