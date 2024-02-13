use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use metal::symbolic::BigExpression;
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use petgraph::visit::EdgeRef;

use crate::{
    op::{InputTensor, Operator},
    prelude::{metal::get_buffer_from_tensor, *},
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

impl<T> QuantizedMatmul<T> {
    fn new(device: Device, queue: CommandQueue) -> Self {
        Self {
            matvec_pipeline: compile_function("mkernel", "
using namespace metal;
#define QK8_0 32
#define NB_Q8_0 8
typedef struct {
    half    d;         // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

kernel void mkernel(
    device const  void* src0 [[buffer(0)]], // Quantized 2D matrix
    device const half* src1 [[buffer(1)]], // Float src vector
    device       half* dst [[buffer(2)]], // Float dest vector
    constant   int64_t & src_vec_size [[buffer(3)]], // Matrix n cols (src vector size) (Must be >= 32)
    constant   int64_t & dest_vec_size [[buffer(4)]], // Matrix n rows (dest vector size) (Must be >= 4)
    constant   int64_t & mat_batch_stride [[buffer(5)]], // Matrix batch stride
    constant   int64_t & vec_batch_stride [[buffer(6)]], // Vector batch stride
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint  thread_index_in_simdgroup[[thread_index_in_simdgroup]],
    uint  simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]] // 2 simdgroups in a threadgroup
) {
    const int num_rows = 4;
    const int num_simdgroups_per_threadgroup = 2;
    const int quant_width = 32;

    const int num_quants_per_row = src_vec_size / 32; // Number of quants per row

    // This is the first row the simdgroup will work on (each simdgroup handles a block of 4 rows)
    const int first_row = (threadgroup_position_in_grid.x * num_simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * num_rows;

    // Offset in number of quant blocks
    device const block_q8_0* x = ((device const block_q8_0*)src0) + (first_row * num_quants_per_row); // Add batch offset here
    device const half* y = src1 + (threadgroup_position_in_grid.z * vec_batch_stride); // Add batch offset here
    dst += (threadgroup_position_in_grid.z * dest_vec_size);

    // thread-local cache of vector values to work on. This thread must only work on 8 at a time
    half yl[8];
    // thread-local cache of 4 row sums
    half sumf[num_rows] = {0.h};

    const int ix = thread_index_in_simdgroup / 4;
    const int il = thread_index_in_simdgroup % 4;

    y += thread_index_in_simdgroup * 8;

    // each thread in a SIMD group deals with 8 quants at a time
    // we start at 0-7 (ix) depending on the simdgroup index, and jump 8 indexes each time
    for (int ib = ix; ib < num_quants_per_row; ib += 8) { // ib: current column position
        // Load vector values into the cache
        #pragma unroll(8)
        for (int i = 0; i < 8; ++i) {
            yl[i] = y[i];
        }

        // Loop through 4 matrix rows
        #pragma unroll(4)
        for (int row = 0; row < 4; ++row) {
            // Get pointer to matrix data
            device const int8_t* qs = x[ib + row * num_quants_per_row].qs + il * 8;
            half sumq = 0.h; // Partial sum
            // Loop through 8 columns
            #pragma unroll(8)
            for (int iq = 0; iq < 8; ++iq) {
                sumq += (half)qs[iq] * yl[iq]; // Multiply int with vector value (auto converts to float?)
            }
            sumf[row] += sumq * x[ib + row * num_quants_per_row].d; // multiply by delta (scaling factor)
        }
        y += 256; // Jump by 256
    }

    // each simdgroup is responsible for saving 4 final vector values (n rows)
    #pragma unroll(4)
    for (int row = 0; row < num_rows; ++row) {
        const half tot = simd_sum(sumf[row]);
        if (thread_index_in_simdgroup == 0 && first_row + row < dest_vec_size) {
            dst[first_row + row] = tot;
        }
    }
}
", &device),
            matmul_pipeline: compile_function("mkernel", "
kernel void mkernel() {}", &device),
//             matmul_pipeline: compile_function("mkernel", "
// using namespace metal;

// #define QK8_0 32
// typedef struct {
//     half    d;         // delta
//     int8_t  qs[QK8_0]; // quants
// } block_q8_0;

// #define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
// #define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
// #define BLOCK_SIZE_K 32
// #define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
// #define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
// #define THREAD_PER_BLOCK 128
// #define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
// #define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
// #define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
// #define SG_MAT_ROW 8

// void dequantize_q8_0(device const block_q8_0 *xb, short il, thread half4x4 & reg) {
//     device const int8_t * qs = ((device const int8_t *)xb->qs);
//     const half d = xb->d;

//     for (int i = 0; i < 16; i++) {
//         reg[i/4][i%4] = (qs[i + 16*il] * d);
//     }
// }

// // each block_q contains 16*nl weights
// kernel void mkernel(
//     device const  uchar * src0 [[buffer(0)]], // Q8_0 matrix
//     device const  uchar * src1 [[buffer(1)]], // Float matrix
//     device        float * dst [[buffer(2)]],
//     constant   int64_t & ne00 [[buffer(3)]], // Matrix n cols (src vector size)
//     constant   int64_t & ne01 [[buffer(4)]], // Matrix n rows (dest vector size)
//     constant   int64_t & ne02 [[buffer(5)]], // Batch of some sort
//     constant   int64_t & ne10 [[buffer(6)]], // Always equal to ne00
//     constant   int64_t & ne12 [[buffer(7)]], // Batch of some other sort
//     constant   int64_t & ne0 [[buffer(8)]], // Always equal to ne01
//     constant   int64_t & ne1 [[buffer(9)]], // Always 1
//     constant   uint    & r2 [[buffer(10)]], // ne12 / ne02
//     constant   uint    & r3 [[buffer(11)]], // ne13 / ne03
//     threadgroup   uchar * shared_memory [[threadgroup(0)]],
//     uint3                 tgpig[[threadgroup_position_in_grid]],
//     uint                  tiitg[[thread_index_in_threadgroup]],
//     uint                  sgitg[[simdgroup_index_in_threadgroup]]
// ) {
//     threadgroup half  * sa = (threadgroup half  *)(shared_memory);
//     threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

//     const uint r0 = tgpig.y;
//     const uint r1 = tgpig.x;
//     const uint im = tgpig.z;

//     // if this block is of 64x32 shape or smaller
//     short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
//     short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

//     // a thread shouldn't load data outside of the matrix
//     short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
//     short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

//     simdgroup_half8x8  ma[4];
//     simdgroup_float8x8 mb[2];
//     simdgroup_float8x8 c_res[8];
//     for (int i = 0; i < 8; i++){
//         c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
//     }

//     short il = (tiitg % THREAD_PER_ROW);

//     const uint i12 = im%ne12;
//     const uint i13 = im/ne12;

//     uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);
//     ushort offset1 = il/2;

//     device const block_q8_0 * x = (device const block_q8_0 *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
//     device const float   * y = (device const float   *)(src1
//         + nb12 * im
//         + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
//         + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

//     for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
//         // load data and store to threadgroup memory
//         half4x4 temp_a;
//         dequantize_q8_0(x, il, temp_a);
//         threadgroup_barrier(mem_flags::mem_threadgroup);

//         #pragma unroll(16)
//         for (int i = 0; i < 16; i++) {
//             *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
//             +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
//             +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
//         }

//         *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

//         il = (il + 2 < 2) ? il + 2 : il % 2;
//         x  = (il < 2) ? x + (2+2-1)/2 : x;
//         y += BLOCK_SIZE_K;

//         threadgroup_barrier(mem_flags::mem_threadgroup);

//         // load matrices from threadgroup memory and conduct outer products
//         threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
//         threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

//         #pragma unroll(4)
//         for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
//             #pragma unroll(4)
//             for (int i = 0; i < 4; i++) {
//                 simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
//             }
//             simdgroup_barrier(mem_flags::mem_none);
//             #pragma unroll(2)
//             for (int i = 0; i < 2; i++) {
//                 simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
//             }

//             lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
//             lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

//             #pragma unroll(8)
//             for (int i = 0; i < 8; i++){
//                 simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
//             }
//         }
//     }

//     if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
//         device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
//                                 + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
//         for (int i = 0; i < 8; i++) {
//             simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
//         }
//     } else {
//         // block is smaller than 64x32, we should avoid writing data outside of the matrix
//         threadgroup_barrier(mem_flags::mem_threadgroup);
//         threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
//                                         + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
//         for (int i = 0; i < 8; i++) {
//             simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
//         }

//         threadgroup_barrier(mem_flags::mem_threadgroup);

//         device float * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
//         if (sgitg == 0) {
//             for (int i = 0; i < n_rows; i++) {
//                 for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
//                     *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
//                 }
//             }
//         }
//     }
// }", &device),
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

pub struct MetalCompilerQ<T>(Vec<NodeIndex>, PhantomData<T>);

impl<T> MetalCompilerQ<T> {
    pub fn new(weights: Vec<NodeIndex>) -> Self {
        Self(weights, Default::default())
    }
}

impl<T: MetalFloat + Default> Compiler for MetalCompilerQ<T> {
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
        // Modify matmul dequantize functions
        // Find matmuls directly downstream of weights
        for weight in downstream(&weight_ids, graph) {
            for (target, (input_ind, _, _)) in graph
                .graph
                .edges_directed(weight, petgraph::Direction::Outgoing)
                .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
                .filter(|(e, (i, _, _))| {
                    *i == 1
                        && graph
                            .graph
                            .node_weight(*e)
                            .unwrap()
                            .as_any()
                            .is::<super::matmul::Matmul<T>>()
                })
                .collect::<Vec<_>>()
            {
                if let Some(matmul_node) = graph.graph.node_weight_mut(target) {
                    // Input ind is a quantized tensor
                    *matmul_node =
                        Box::new(QuantizedMatmul::<T>::new(device.clone(), queue.clone()))
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

    #[repr(packed)]
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
        let mat_data: Vec<i8> = (0..(32 * 4)).map(|_| rng.gen_range(0..5)).collect();
        let vec_data = random_vec_rng(32, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<4, 32>>();
        let vec = cx.tensor::<R1<32>>().set(vec_data.clone());
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

        cx.compile(MetalCompilerQ::<f16>::new(vec![weights.id]), &mut out);
        cx.execute();

        let cpu = Cpu::default();
        let d_a = cpu
            .tensor_from_vec(
                mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
                (DConst::<4>, DConst::<32>),
            )
            .to_dtype::<f16>();
        let d_b = cpu
            .tensor_from_vec(vec_data, (DConst::<32>,))
            .to_dtype::<f16>();
        let d_c = d_b.matmul(d_a.permute());
        assert_close_precision(&out.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
    }

    #[test]
    fn test_quantized_matmul() {
        let mut rng = thread_rng();
        let mat_data: Vec<i8> = (0..(32 * 4)).map(|_| rng.gen_range(0..5)).collect();
        let inp_mat_data = random_vec_rng(32 * 2, &mut rng);
        let mut cx = Graph::new();
        let weights = cx.tensor::<R2<4, 32>>();
        let inp_mat = cx.tensor::<R2<2, 32>>().set(inp_mat_data.clone());
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

        cx.compile(MetalCompilerQ::<f16>::new(vec![weights.id]), &mut out);
        cx.execute();

        let cpu = Cpu::default();
        let d_a = cpu
            .tensor_from_vec(
                mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
                (DConst::<4>, DConst::<32>),
            )
            .to_dtype::<f16>();
        let d_b = cpu
            .tensor_from_vec(inp_mat_data, (DConst::<2>, DConst::<32>))
            .to_dtype::<f16>();
        let d_c = d_b.matmul(d_a.permute());
        assert_close_precision(&out.data(), &d_c.to_dtype::<f32>().as_vec(), 2);
    }
}
