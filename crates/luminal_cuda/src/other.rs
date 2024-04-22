use std::{marker::PhantomData, sync::Arc};

use luminal_cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

use fmt_derive::Debug;
use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    binary::CudaSub,
    compile_and_load_kernel, constant,
    prim::{CudaAdd, CudaContiguous, CudaSumReduce},
    CudaData, CudaFloat,
};

#[derive(Clone, Debug)]
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

        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Debug, Default)]
pub struct ARangeCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for ARangeCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig_one = constant::<T>(1.);
        let contig1 = unary::<CudaContiguous<T>>(contig_one.clone());
        let sum_reduce =
            unary::<CudaSumReduce<T>>(unary::<CudaContiguous<T>>(unary::<CudaContiguous<T>>(
                unary::<CudaContiguous<T>>(contig1.clone()),
            )));
        let sub = binary::<CudaSub<T>>(sum_reduce.clone(), constant::<T>(1.));
        let mut s1 = sub.clone().search(graph);
        let neg_one = constant::<T>(-1.);
        let add = binary::<CudaAdd<T>>(sum_reduce, neg_one.clone());
        let mut s2 = add.clone().search(graph);

        while s1.next_match() || s2.next_match() {
            let s = if s1.matched { &s1 } else { &s2 };
            let arange_amount = {
                let sh = graph
                    .edges_connecting(s.get(&contig_one), s.get(&contig1))
                    .next()
                    .unwrap()
                    .weight()
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
            let fin = if s1.matched {
                s1.get(&sub)
            } else {
                s2.get(&add)
            };
            move_outgoing_edge(fin, arange_op, graph);
            graph.remove_node(fin);
            s.try_delete();
        }
    }
}
