use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use metal::symbolic::BigExpression;
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use petgraph::visit::EdgeRef;

use crate::{
    op::{InputTensor, Operator},
    prelude::{
        metal::{binary::MetalGather, get_buffer_from_tensor},
        *,
    },
};

use super::{compile_function, SetInt};

/// Multiplies a BxMxK matrix with a KxN matrix, resulting in a BxMxN matrix. This expects the first input to be a quantized 2D matrix
#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct QuantizedMatmul<T> {
    matvec_pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> QuantizedMatmul<T> {
    fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        Self {
            matvec_pipeline: compile_function("mkernel", &format!("
using namespace metal;
#define QK8_0 32
#define NB_Q8_0 8
typedef struct {{
    half    d;         // delta
    int8_t  qs[QK8_0]; // quants
}} block_q8_0;

kernel void mkernel(
    device block_q8_0* x [[buffer(0)]], // Quantized 2D matrix
    device {type_name}* y [[buffer(1)]], // Float src vector
    device {type_name}* dst [[buffer(2)]], // Float dest vector
    constant int64_t & src_vec_size [[buffer(3)]], // Matrix n cols (src vector size) (Must be >= 32)
    constant int64_t & dest_vec_size [[buffer(4)]], // Matrix n rows (dest vector size) (Must be >= 4)
    constant int64_t & mat_batch_stride [[buffer(5)]], // Matrix batch stride
    constant int64_t & vec_batch_stride [[buffer(6)]], // Vector batch stride
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint  thread_index_in_simdgroup[[thread_index_in_simdgroup]],
    uint  simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]] // 2 simdgroups in a threadgroup
) {{
    const int num_rows = 4;
    const int num_simdgroups_per_threadgroup = 2;
    const int quant_width = 32;

    const int num_quants_per_row = src_vec_size / 32; // Number of quants per row

    // This is the first row the simdgroup will work on (each simdgroup handles a block of 4 rows)
    const int first_row = (threadgroup_position_in_grid.x * num_simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * num_rows;

    // Offsets
    x += first_row * num_quants_per_row + threadgroup_position_in_grid.z * (mat_batch_stride / 32);
    y += threadgroup_position_in_grid.z * vec_batch_stride;
    dst += (threadgroup_position_in_grid.z * dest_vec_size);

    // thread-local cache of vector values to work on. This thread must only work on 8 at a time
    {type_name} yl[8];
    // thread-local cache of 4 row sums
    float sumf[num_rows] = {{0.f}};

    const int ix = thread_index_in_simdgroup / 4;
    const int il = thread_index_in_simdgroup % 4;

    y += thread_index_in_simdgroup * 8;

    // each thread in a SIMD group deals with 8 quants at a time
    // we start at 0-7 (ix) depending on the simdgroup index, and jump 8 indexes each time
    for (int ib = ix; ib < num_quants_per_row; ib += 8) {{ // ib: current column position
        // Load vector values into the cache
        for (int i = 0; i < 8; ++i) {{
            yl[i] = y[i];
        }}

        // Loop through 4 matrix rows
        for (int row = 0; row < 4; ++row) {{
            // Get pointer to matrix data
            device const int8_t* qs = x[ib + row * num_quants_per_row].qs + il * 8;
            float sumq = 0.f; // Partial sum
            // Loop through 8 columns
            for (int iq = 0; iq < 8; ++iq) {{
                sumq += qs[iq] * yl[iq]; // Multiply int with vector value (auto converts to float?)
            }}
            sumf[row] += sumq * x[ib + row * num_quants_per_row].d; // multiply by delta (scaling factor)
        }}
        y += 256; // Jump by 256
    }}

    // each simdgroup is responsible for saving 4 final vector values (n rows)
    for (int row = 0; row < num_rows; ++row) {{
        const float tot = simd_sum(sumf[row]);
        if (thread_index_in_simdgroup == 0 && first_row + row < dest_vec_size) {{
            dst[first_row + row] = ({type_name})tot;
        }}
    }}
}}
"), &device),
            queue,
            device,
            _phantom: Default::default(),
        }
    }
}

impl<T> MetalKernel for QuantizedMatmul<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        let m = input_shapes[0].shape()[input_shapes[0].len() - 2].clone();
        let n = input_shapes[1].shape()[input_shapes[1].len() - 1].clone();
        let batch_size = input_shapes[0]
            .shape()
            .into_iter()
            .take(input_shapes[0].len() - 2)
            .product::<BigExpression>()
            .max(BigExpression::from(1));
        vec![batch_size * m * n * size_of::<T>()]
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        assert!(
            !inputs[1].1.is_contiguous(),
            "Weight matrix must be column-major"
        );
        let (a_shape, b_shape) = (
            inputs[0]
                .1
                .shape()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
            inputs[1]
                .1
                .shape()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
        );
        let a_dims = a_shape.len();
        let m = a_shape[a_dims - 2];
        let batch_size = a_shape.iter().take(a_dims - 2).product::<usize>().max(1);
        let b_dims = b_shape.len();
        let k = b_shape[b_dims - 2];
        let n = b_shape[b_dims - 1];

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        if batch_size == 1 {
            // Matvec
            encoder.set_compute_pipeline_state(&self.matvec_pipeline);
            encoder.set_buffer(0, Some(inputs[1].0), 0); // Matrix
            encoder.set_buffer(1, Some(inputs[0].0), 0); // Vector
            encoder.set_buffer(2, Some(output_buffers[0]), 0); // Dest vector
            encoder.set_i64(3, k as i64); // Src vec size
            encoder.set_i64(4, n as i64); // Dest vec size
            encoder.set_i64(5, 0); // Matrix batch stride
            encoder.set_i64(6, k as i64); // Vector batch stride
            encoder.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(8) as u64, 1, m as u64),
                MTLSize::new(8, 8, 1),
            );
        } else {
            todo!()
        }
        encoder.end_encoding();
    }
}

impl<T: 'static + Clone> Operator for QuantizedMatmul<T> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup command queue / command buffer / encoder
            let command_buffer = self.queue.new_command_buffer();

            let (a_shape, b_shape) = (inp[0].1.shape(), inp[1].1.shape());
            let n = b_shape[1].to_usize().unwrap();
            let (batch_size, m) = if a_shape.len() == 3 {
                (
                    a_shape[0].to_usize().unwrap(),
                    a_shape[1].to_usize().unwrap(),
                )
            } else {
                (0, a_shape[0].to_usize().unwrap())
            };

            let out = self.device.new_buffer(
                (batch_size * m * n * std::mem::size_of::<T>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(
                &[
                    (get_buffer_from_tensor(&inp[0].0), inp[0].1),
                    (get_buffer_from_tensor(&inp[1].0), inp[1].1),
                ],
                command_buffer,
                &[],
                &[&out],
            );

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(out)]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct QuantizedGather<T> {
    pipeline: ComputePipelineState,
    device: Device,
    queue: CommandQueue,
    embed_dim: usize,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> QuantizedGather<T> {
    fn new(device: Device, queue: CommandQueue, embed_dim: usize) -> Self {
        let type_name = T::type_name();
        Self {pipeline: compile_function("metal_gather", &format!(
            "
#include <metal_stdlib>
using namespace metal;
#define QK8_0 32
typedef struct {{
    half    d;         // delta
    int8_t  qs[QK8_0]; // quants
}} block_q8_0;

kernel void metal_gather(device float *inp [[buffer(0)]], device block_q8_0 *weights [[buffer(1)]], device {type_name} *out [[buffer(2)]], device int& n_embeddings [[buffer(3)]], device int& embedding_dim [[buffer(4)]], uint2 idx [[thread_position_in_grid]]) {{
    if (idx.x < n_embeddings && idx.y < embedding_dim) {{
        int block_idx = ((int)inp[idx.x] * embedding_dim + idx.y) / QK8_0;
        out[idx.x * embedding_dim + idx.y] = weights[block_idx].qs[idx.y % QK8_0] * weights[block_idx].d;
    }}
}}"), &device), device, embed_dim, queue, _phantom: Default::default()}
    }
}

impl<T: MetalFloat> Operator for QuantizedGather<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Setup buffers
            let indexes = tensors[0]
                .0
                .borrowed()
                .data
                .as_any()
                .downcast_ref::<Vec<f32>>()
                .unwrap();
            let index_buffer = self.device.new_buffer_with_data(
                unsafe { std::mem::transmute(indexes.as_ptr()) },
                (indexes.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

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
            encoder.set_buffer(1, Some(get_buffer_from_tensor(&tensors[1].0)), 0);
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

            vec![Tensor::new(out)]
        })
    }
}

#[derive(Default)]
pub struct MetalQuantizedCompiler<T>(Vec<NodeIndex>, PhantomData<T>);

impl<T> MetalQuantizedCompiler<T> {
    pub fn new<To: ToIds>(weights: To) -> Self {
        Self(weights.to_ids(), Default::default())
    }
}

impl<T: MetalFloat + Default> Compiler for MetalQuantizedCompiler<T> {
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut remap: To) {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let mut weight_ids = self.0.clone();
        let mut local_remap = remap.to_ids_mut();
        for w in &mut weight_ids {
            local_remap.push(w);
        }
        // Normal metal compilation
        graph.compile(
            <(
                super::prim::PrimitiveCompiler<T>,
                super::SpecialOpsCompiler<T>,
                super::other::CopyCompiler<T>,
                super::other::ContiguousElimination<T>,
                super::elementwise_fusion::ElementwiseFusionCompiler<T>,
            )>::default(),
            &mut local_remap,
        );
        // Modify ops directly downstream of weights
        for weight in downstream(&weight_ids, graph) {
            for (target, (inp_ind, _, _)) in graph
                .graph
                .edges_directed(weight, petgraph::Direction::Outgoing)
                .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
                .collect::<Vec<_>>()
            {
                assert_eq!(
                    inp_ind, 1,
                    "Quantized weight {target:?} is the wrong input!",
                );
                let op_node = graph.graph.node_weight_mut(target).unwrap();
                if let Some(gather) = op_node.as_any().downcast_ref::<MetalGather<T>>() {
                    *op_node = Box::new(QuantizedGather::<T>::new(
                        device.clone(),
                        queue.clone(),
                        gather.embed_dim,
                    ));
                } else if op_node.as_any().is::<super::matmul::Matmul<T>>() {
                    *op_node = Box::new(QuantizedMatmul::<T>::new(device.clone(), queue.clone()));
                } else {
                    panic!("Quantized weight {target:?} is an input to a node that isn't a matmul or gather!");
                }
            }
        }
        // Finish normal metal compilation
        graph.compile(super::BufferCompilers::default(), &mut remap);
    }
}

#[cfg(test)]
mod tests {
    use metal_rs::{Device, MTLResourceOptions};
    use rand::{thread_rng, Rng};

    #[repr(C, packed)]
    struct BlockQ8_0 {
        _d: f16,
        _qs: [i8; 32],
    }

    fn quantized_buffer(weights: &[BlockQ8_0], dev: &Device) -> crate::prelude::Tensor {
        let buffer = dev.new_buffer_with_bytes_no_copy(
            weights.as_ptr() as *mut _,
            (weights.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        crate::prelude::Tensor {
            data: Box::new(buffer),
        }
    }

    crate::test_imports!();
    #[test]
    fn test_quantized_matvec() {
        let mut rng = thread_rng();
        let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
        let vec_data = random_vec_rng(1024, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<512, 1024>>();
        let vec = cx.tensor::<R1<1024>>().set(vec_data.clone());
        let mut out = vec.matmul(weights.permute()).retrieve();

        // "Load" weights in 8bit
        let blocks = mat_data
            .chunks_exact(32)
            .map(|chunk| {
                let mut array = [0; 32];
                for (i, n) in chunk.iter().enumerate() {
                    array[i] = *n;
                }
                BlockQ8_0 {
                    _d: f16::from_f32(1.0),
                    _qs: array,
                }
            })
            .collect::<Vec<_>>();
        let dev = Device::system_default().unwrap();
        cx.tensors
            .insert((weights.id, 0), quantized_buffer(&blocks, &dev));

        cx.compile(
            MetalQuantizedCompiler::<f32>::new(vec![weights.id]),
            &mut out,
        );
        cx.execute();

        let mut cx1 = Graph::new();
        let weights = cx1
            .tensor::<R2<512, 1024>>()
            .set(mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>());
        let vec = cx1.tensor::<R1<1024>>().set(vec_data);
        let out_32 = vec.matmul(weights.permute()).retrieve();
        cx1.execute();

        assert_close(&out.data(), &out_32.data());
    }

    #[test]
    fn test_quantized_matmul() {
        let mut rng = thread_rng();
        let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
        let inp_mat_data = random_vec_rng(1024 * 16, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<512, 1024>>();
        let inp_mat = cx.tensor::<R2<16, 1024>>().set(inp_mat_data.clone());
        let mut out = inp_mat.matmul(weights.permute()).retrieve();

        // "Load" weights in 8bit
        let blocks = mat_data
            .chunks_exact(32)
            .map(|chunk| {
                let mut array = [0; 32];
                for (i, n) in chunk.iter().enumerate() {
                    array[i] = *n;
                }
                BlockQ8_0 {
                    _d: f16::from_f32(1.0),
                    _qs: array,
                }
            })
            .collect::<Vec<_>>();
        let dev = Device::system_default().unwrap();
        cx.tensors
            .insert((weights.id, 0), quantized_buffer(&blocks, &dev));

        cx.compile(
            MetalQuantizedCompiler::<f32>::new(vec![weights.id]),
            &mut out,
        );
        cx.execute();

        let cpu = Cpu::default();
        let d_a = cpu.tensor_from_vec(
            mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
            (DConst::<512>, DConst::<1024>),
        );
        let d_b = cpu.tensor_from_vec(inp_mat_data, (DConst::<16>, DConst::<1024>));
        let d_c = d_b.matmul(d_a.permute());
        assert_close(&out.data(), &d_c.as_vec());
    }
}
