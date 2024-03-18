use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use petgraph::visit::EdgeRef;

use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
    shape::symbolic::BigExpression,
};

use crate::{
    binary::MetalGather, get_buffer_from_tensor, MetalBuffer, MetalFloat, MetalKernel,
    MetalKernelWrapper,
};

use super::{compile_function, SetInt};

/// Multiplies a BxMxK matrix with a KxN matrix, resulting in a BxMxN matrix. This expects the first input to be a quantized 2D matrix
#[derive(LuminalEqFalse, LuminalPrint, Clone)]
pub struct QuantizedMatmul<T> {
    matmul_pipeline: ComputePipelineState,
    matvec_pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> QuantizedMatmul<T> {
    fn new(device: Device, queue: CommandQueue) -> Self {
        let type_name = T::type_name();
        Self {
            matmul_pipeline: compile_function("matmul", &format!("
using namespace metal;
typedef struct {{
    half    d;         // delta
    int8_t  qs[32]; // quants
}} block_q8_0;

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

void dequantize_q8_0(device const block_q8_0 *xb, short il, thread half4x4 & reg) {{
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    for (int i = 0; i < 16; i++) {{
        reg[i/4][i%4] = (qs[i + 16*il] * d);
    }}
}}

kernel void matmul(
    device const  uchar * src0 [[buffer(0)]],
    device const  uchar * src1 [[buffer(1)]],
    device        {type_name} * dst [[buffer(2)]],
    constant    int64_t & ne00 [[buffer(3)]], // k
    constant    int64_t & ne02 [[buffer(4)]], // Always 1
    constant   uint64_t & nb01 [[buffer(5)]], // k * 1.0625 (avg bytes per weight)
    constant   uint64_t & nb02 [[buffer(6)]], // k * 1.0625 * n
    constant    int64_t & ne12 [[buffer(7)]], // Always 1
    constant   uint64_t & nb10 [[buffer(8)]], // bytes of dtype
    constant   uint64_t & nb11 [[buffer(9)]], // bytes of dtype * k
    constant   uint64_t & nb12 [[buffer(10)]], // bytes of dtype * k * m
    constant    int64_t & ne0 [[buffer(11)]], // n
    constant    int64_t & ne1 [[buffer(12)]], // m
    constant       uint & r2 [[buffer(13)]], // 1
    constant       uint & r3 [[buffer(14)]], // 1
    threadgroup   uchar * shared_memory [[threadgroup(0)]],
    uint3                 tgpig[[threadgroup_position_in_grid]],
    uint                  tiitg[[thread_index_in_threadgroup]],
    uint                  sgitg[[simdgroup_index_in_threadgroup]]
) {{

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup {type_name} * sb = (threadgroup {type_name} *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    const short nl = 2;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    // Simdgroup a b and c mat storage
    simdgroup_half8x8  ma[4];
    simdgroup_{type_name}8x8 mb[2];
    simdgroup_{type_name}8x8 c_res[8]; // Accumulate into c so init to 0
    for (int i = 0; i < 8; i++){{
        c_res[i] = make_filled_simdgroup_matrix<{type_name}, 8>(0.0);
    }}

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);
    ushort offset1 = il/nl;

    device const block_q8_0 * x = (device const block_q8_0 *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const {type_name}   * y = (device const {type_name}   *)(src1
        + nb12 * im
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {{
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_q8_0(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {{
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }}

        *(threadgroup {type_name}2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device {type_name}2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup {type_name} * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {{
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {{
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }}
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {{
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }}

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){{
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }}
        }}
    }}

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {{
        device {type_name} * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (int i = 0; i < 8; i++) {{
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }}
    }} else {{
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup {type_name} * temp_str = ((threadgroup {type_name} *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {{
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device {type_name} * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg == 0) {{
            for (int i = 0; i < n_rows; i++) {{
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {{
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }}
            }}
        }}
    }}
}}"), &device),
            matvec_pipeline: compile_function("matvec", &format!("
using namespace metal;
typedef struct {{
    half    d;         // delta
    int8_t  qs[32]; // quants
}} block_q8_0;

kernel void matvec(
    device block_q8_0* x [[buffer(0)]], // Quantized 2D matrix
    device {type_name}* y [[buffer(1)]], // Float src vector
    device {type_name}* dst [[buffer(2)]], // Float dest vector
    constant uint & src_vec_size [[buffer(3)]], // Matrix n cols (src vector size) (Must be >= 32)
    constant uint & dest_vec_size [[buffer(4)]], // Matrix n rows (dest vector size) (Must be >= 4)
    constant uint & mat_batch_stride [[buffer(5)]], // Matrix batch stride
    constant uint & vec_batch_stride [[buffer(6)]], // Vector batch stride
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
    // x += first_row * num_quants_per_row + threadgroup_position_in_grid.z * (mat_batch_stride / 32); // Batch offset
    x += first_row * num_quants_per_row; // No batch offset
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
        let b_dims = b_shape.len();
        let batch_size = a_shape.iter().take(a_dims - 2).product::<usize>().max(1);
        let m = a_shape[a_dims - 2];
        let k = b_shape[b_dims - 2];
        let n = b_shape[b_dims - 1];

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        if m == 1 && batch_size == 1 {
            // Matvec
            encoder.set_compute_pipeline_state(&self.matvec_pipeline);
            encoder.set_buffer(0, Some(inputs[1].0), 0); // Matrix
            encoder.set_buffer(1, Some(inputs[0].0), 0); // Vector
            encoder.set_buffer(2, Some(output_buffers[0]), 0); // Dest vector
            encoder.set_u32(3, k as u32); // Src vec size
            encoder.set_u32(4, n as u32); // Dest vec size
            encoder.set_u32(5, 0); // Matrix batch stride
            encoder.set_u32(6, k as u32); // Vector batch stride
            encoder.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                MTLSize::new(8, 8, 1),
            );
        } else {
            // Matmul
            encoder.set_compute_pipeline_state(&self.matmul_pipeline);
            encoder.set_buffer(0, Some(inputs[1].0), 0); // Weight matrix
            encoder.set_buffer(1, Some(inputs[0].0), 0); // Input matrix
            encoder.set_buffer(2, Some(output_buffers[0]), 0); // Dest matrix
            encoder.set_i64(3, k as i64);
            encoder.set_i64(4, 1);
            encoder.set_i64(5, (k as f32 * 1.0625) as i64);
            encoder.set_i64(6, (k as f32 * 1.0625 * n as f32) as i64);
            encoder.set_i64(7, 1);
            encoder.set_i64(8, size_of::<T>() as i64);
            encoder.set_i64(9, (size_of::<T>() * k) as i64);
            encoder.set_i64(10, (size_of::<T>() * k * m) as i64);
            encoder.set_i64(11, n as i64);
            encoder.set_i64(12, m as i64);
            encoder.set_u32(13, 1);
            encoder.set_u32(14, 1);
            encoder.set_threadgroup_memory_length(0, 8192);
            encoder.dispatch_thread_groups(
                MTLSize::new((m as u64 + 31) / 32, (n as u64 + 63) / 64, 1),
                MTLSize::new(128, 1, 1),
            );
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

            vec![Tensor::new(MetalBuffer(out))]
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

            vec![Tensor::new(MetalBuffer(out))]
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
    type Output = ();
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
                super::elementwise_fusion::ElementwiseFusionCompiler<T>,
            )>::default(),
            &mut local_remap,
        );
        // Modify ops directly downstream of weights
        for weight in downstream(&weight_ids, graph) {
            for (target, (inp_ind, _, _)) in graph
                .edges_directed(weight, petgraph::Direction::Outgoing)
                .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
                .collect::<Vec<_>>()
            {
                assert_eq!(
                    inp_ind, 1,
                    "Quantized weight {target:?} is the wrong input!",
                );
                let op_node = graph.node_weight_mut(target).unwrap();
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
    use dfdx::{
        tensor::TensorFromVec,
        tensor_ops::{PermuteTo, TryMatMul},
    };
    use luminal::{
        prelude::*,
        tests::{assert_close, assert_close_precision, random_vec_rng},
    };
    use metal_rs::{Device, MTLResourceOptions};
    use rand::{thread_rng, Rng};

    use crate::{MetalBuffer, MetalQuantizedCompiler};

    #[repr(C, packed)]
    struct BlockQ8_0 {
        _d: f16,
        _qs: [i8; 32],
    }

    fn quantized_buffer(weights: &[BlockQ8_0], dev: &Device) -> Tensor {
        let buffer = dev.new_buffer_with_data(
            weights.as_ptr() as *mut _,
            std::mem::size_of_val(weights) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Tensor {
            data: Box::new(MetalBuffer(buffer)),
        }
    }

    #[test]
    fn test_quantized_matvec() {
        let mut rng = thread_rng();
        let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
        let vec_data = random_vec_rng(1024, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<512, 1024>>().keep();
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
            .set(mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>())
            .keep();
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
        let weights = cx.tensor::<R2<512, 1024>>().keep();
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

        let cpu = dfdx::tensor::Cpu::default();
        let d_a = cpu.tensor_from_vec(
            mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
            (dfdx::shapes::Const::<512>, dfdx::shapes::Const::<1024>),
        );
        let d_b = cpu.tensor_from_vec(
            inp_mat_data,
            (dfdx::shapes::Const::<16>, dfdx::shapes::Const::<1024>),
        );
        let d_c = d_b.matmul(d_a.permute());
        assert_close(&out.data(), &d_c.as_vec());
    }

    #[test]
    fn test_quantized_matmul_fp16() {
        let mut rng = thread_rng();
        let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
        let inp_mat_data = random_vec_rng(1024 * 16, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<512, 1024>>().keep();
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
            MetalQuantizedCompiler::<f16>::new(vec![weights.id]),
            &mut out,
        );
        cx.execute();

        let cpu = dfdx::tensor::Cpu::default();
        let d_a = cpu.tensor_from_vec(
            mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
            (dfdx::shapes::Const::<512>, dfdx::shapes::Const::<1024>),
        );
        let d_b = cpu.tensor_from_vec(
            inp_mat_data,
            (dfdx::shapes::Const::<16>, dfdx::shapes::Const::<1024>),
        );
        let d_c = d_b.matmul(d_a.permute());
        assert_close_precision(&out.data(), &d_c.as_vec(), 0);
        // This is imprecise currently because we accumulate in fp16 in the matmul. TODO: accumulate in fp32 and convert before saving to dest
    }
}
