use std::{marker::PhantomData, sync::Arc};

use luminal_cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaDevice, CudaFunction, DevicePtr, DevicePtrMut},
};
use rustc_hash::FxHashMap;

use crate::{
    prim::{CudaMul, CudaSumReduce},
    CudaData, CudaFloat,
};
use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
};

#[derive(Debug, Default)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

impl<T> Compiler for ElementwiseFusionCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, remap: To) {
        // Track fused ops
        let mut fused_ops = FxHashMap::<NodeIndex, (String, Vec<ShapeTracker>)>::default();
    }
}

#[derive(LuminalPrint, Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<CudaFunction>,
    dyn_map: *const FxHashMap<char, usize>,
    dyn_chars: Vec<char>,
    equantion: String,
    device: Arc<CudaDevice>,
    input_views: Vec<ShapeTracker>,
    _phantom: PhantomData<T>,
}
impl<T> PartialEq for FusedElementwiseOp<T> {
    fn eq(&self, other: &Self) -> bool {
        self.equantion == other.equantion && self.input_views == other.input_views
    }
}
