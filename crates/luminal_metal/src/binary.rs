use std::{any::Any, fmt::Debug, marker::PhantomData, mem::size_of, sync::Arc};

use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use rustc_hash::FxHashMap;

use crate::{
    compile_function, constant, get_buffer_from_tensor, get_idx_valid_exps, input_dyn_dims,
    render_dyn_dim_inputs, DispatchNElements, MetalBuffer, MetalFloat, MetalKernel,
    MetalKernelWrapper, SetInt,
};

use super::prim::*;
use luminal::{
    op::{InputTensor, Operator},
    prelude::{petgraph::visit::EdgeRef, *},
};

use super::other::MetalARange;

crate::metal_binary_op!("{} - {}", MetalSub);

#[derive(Debug, Default)]
pub struct MetalSubtractionCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalSubtractionCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let rhs = node();
        let mul = binary::<MetalMul<T>>(rhs.clone(), constant::<T>(-1.));
        let add = unary::<MetalAdd<T>>(mul.clone());
        let mut s = add.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[add.id]) {
                continue;
            }
            let add = s.get(&add);
            let (a, a_edge) = graph
                .edges_directed(add, petgraph::Direction::Incoming)
                .find(|e| e.source() != s.get(&mul))
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (b, b_edge) = graph
                .edges_connecting(s.get(&rhs), s.get(&mul))
                .next()
                .map(|e| (e.source(), e.weight().as_data().unwrap()))
                .unwrap();
            let (_, _, b_final_shape) = graph
                .edges_connecting(s.get(&mul), add)
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            if b_final_shape.is_reshaped() {
                continue;
            }
            let sub = graph
                .add_op(MetalSub::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ))
                .input(a, a_edge.1, a_edge.2)
                .input(b, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(add, sub, graph);
            remap(add, sub, &mut ids, graph);

            graph.remove_node(add);
            s.try_delete();
        }
    }
}
crate::metal_binary_op!("({type_name})({} == {})", MetalEqual);


#[derive(Debug, Default)]
pub struct MetalEqualCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalEqualCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let one = constant::<T>(1.);
        let (lhs, rhs) = (node(), node());
        let lt1 = binary::<MetalLessThan<T>>(lhs.clone(), rhs.clone());
        let ne = binary::<MetalAdd<T>>(
            lt1.clone(),
            binary::<MetalLessThan<T>>(rhs.clone(), lhs.clone()),
        );
        let eq = binary::<MetalSub<T>>(one, ne);

        let mut s = eq.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[eq.id]) {
                continue;
            }
            let (lhs, rhs) = (s.get(&lhs), s.get(&rhs));
            let eq = s.get(&eq);
            let a_edge = graph
                .edges_connecting(lhs, s.get(&lt1))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let b_edge = graph
                .edges_connecting(rhs, s.get(&lt1))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let equals = graph
                .add_op(MetalEqual::<T>::new(
                    a_edge.2,
                    b_edge.2,
                    dev.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ))
                .input(lhs, a_edge.1, a_edge.2)
                .input(rhs, b_edge.1, b_edge.2)
                .finish();
            move_outgoing_edge(eq, equals, graph);
            remap(eq, equals, &mut ids, graph);

            graph.remove_node(eq);
            s.try_delete();
        }
    }
}

#[derive(Clone)]
pub struct MetalGather<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    pub embed_dim: usize,
    _phantom: PhantomData<T>,
}
impl<T> Debug for MetalGather<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalGather")
    }
}

impl<T: MetalFloat> MetalGather<T> {
    pub fn new(device: Device, queue: CommandQueue, embed_dim: usize) -> Self {
        let type_name = T::type_name();
        Self {pipeline: compile_function("metal_gather", &format!(
            "
#include <metal_stdlib>
using namespace metal;
kernel void metal_gather(device float *inp [[buffer(0)]], device {type_name} *weights [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_embeddings [[buffer(3)]], device int& embedding_dim [[buffer(4)]], uint2 i_ [[thread_position_in_grid]]) {{
    if (i_.x < n_embeddings && i_.y < embedding_dim) {{
        out[i_.x * embedding_dim + i_.y] = weights[(int)inp[i_.x] * embedding_dim + i_.y];
    }}
}}"), &device), device, embed_dim, queue, _phantom: Default::default()}
    }
}

impl<T: MetalFloat> Operator for MetalGather<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let indexes = tensors[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
            let index_buffer = self.device.new_buffer_with_data(
                indexes.as_ptr() as *const _,
                (indexes.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let b_inp = tensors[1]
                .0
                .borrowed()
                .downcast_ref::<MetalBuffer>()
                .unwrap();

            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let out = self.device.new_buffer(
                (indexes.len() * self.embed_dim * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&self.pipeline);

            // Set inputs
            encoder.set_buffer(0, Some(&index_buffer), 0);
            encoder.set_buffer(1, Some(b_inp), 0);
            encoder.set_buffer(2, Some(&out), 0);
            encoder.set_u32(3, indexes.len() as u32);
            encoder.set_u32(4, self.embed_dim as u32);

            // Execute
            encoder.dispatch_threads(
                MTLSize {
                    width: indexes.len() as u64,
                    height: self.embed_dim as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }
}

#[derive(Debug, Default)]
pub struct MetalGatherCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for MetalGatherCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let indexes = node();
        let ind_copy = unary::<MetalCopyToDevice<T>>(indexes.clone());
        let equal = binary::<MetalEqual<T>>(op::<MetalARange<T>>(), ind_copy.clone());
        let embeddings = node();
        let mul = binary::<MetalMul<T>>(embeddings.clone(), equal.clone());
        let sum_reduce = unary::<MetalSumReduce<T>>(mul.clone());
        let mut s = sum_reduce.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sum_reduce.id, embeddings.id, indexes.id]) {
                continue;
            }
            let emb_shape = graph
                .edges_connecting(s.get(&embeddings), s.get(&mul))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let embed_dim = emb_shape.dims()[2].to_usize().unwrap();
            let index_shape = graph
                .edges_connecting(s.get(&indexes), s.get(&ind_copy))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let gather = graph
                .add_op(MetalGather::<T>::new(dev.clone(), queue.clone(), embed_dim))
                .input(s.get(&indexes), 0, index_shape)
                .input(s.get(&embeddings), 0, emb_shape)
                .finish();
            move_outgoing_edge(s.get(&sum_reduce), gather, graph);
            remap(s.get(&sum_reduce), gather, &mut ids, graph);

            graph.remove_node(s.get(&sum_reduce));
            s.try_delete();
        }
    }
}

#[cfg(test)]
mod tests {
    use luminal::{prelude::*, tests::assert_close};

    use crate::MetalCompiler;
    #[test]
    fn test_subtraction() {
        let mut cx = Graph::new();
        let a = cx
            .tensor(10)
            .set(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        let b = cx.tensor(()).set(vec![1.]);
        let mut c = (a - b.expand_to(a.shape)).retrieve();
        let mut d = (-a + b.expand_to(a.shape)).retrieve();

        cx.execute();

        let unopt_c = c.data();
        c.drop();
        let unopt_d = d.data();
        d.drop();

        cx.compile(MetalCompiler::<f16>::default(), (&mut c, &mut d));
        cx.execute();

        assert_close(&unopt_c, &c.data());
        assert_close(&unopt_d, &d.data());
    }
}
