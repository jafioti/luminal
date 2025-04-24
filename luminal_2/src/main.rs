#![allow(clippy::type_complexity)]

// TODO
// Write simdgroup matmul (validated)
// handle thread-level looping (multiple results per thread)
// write optimal simdgroup matmul (4x1)
// write rewrite trajectory for optimal simdgroup matmul
// write flash attention in ir
// write rewrite trajector for flash attention
// Put flattened IR into egglog
// If flattened IR doesn't go into egglog, put nested IR into egglog and write flattening function
// write scoring function (profiling based)
// write rewrite rules in egglog
// get optimal matrix multiplication generated automatically
// get flash attention generated automatically
// get one-layer transformer generated
// get full llm generated
// post update and spread the good word

use std::collections::HashMap;

use itertools::Itertools;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
enum Input {
    Inp(usize), // An input to this scope
    Ref(usize), // A reference to an earlier variable in the scope
}

#[derive(Clone, Debug, Default)]
struct Stack {
    inputs: Vec<Input>,
    instruction: String,
    frames: Vec<StackFrame>,
    kernel_output: bool, // Do we need this output for another kernel?
}

impl std::fmt::Display for Stack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:?}) -> {} | {}{}",
            self.inputs,
            self.instruction,
            self.frames
                .iter()
                .map(
                    |StackFrame {
                         size,
                         input_strides,
                         loop_char,
                         output_stride,
                         ..
                     }| {
                        format!(
                            "{size} ({input_strides:?})[{output_stride}]{}",
                            loop_char.map(|f| format!(" {{{f}}}")).unwrap_or_default()
                        )
                    }
                )
                .join(" | "),
            if self.kernel_output { " -> Out" } else { "" }
        )
    }
}

#[derive(Clone, Debug, Default)]
struct StackFrame {
    size: usize,
    input_strides: Vec<usize>,
    loop_char: Option<char>,
    output_stride: usize,
    reduce: bool,
    thread_loop: bool,
}

#[derive(Debug, Clone)]
struct Kernel {
    code: String,
    grid: (usize, usize, usize),
    threadblock: (usize, usize, usize),
    inputs: Vec<usize>,         // global buffer indexes this kernel uses
    outputs: Vec<usize>,        // buffer sizes this kernel creates
    shared_buffers: Vec<usize>, // sizes of required shared memory buffers
}

#[cfg(target_os = "macos")]
const PRELUDE: &str = "
#include <metal_stdlib>
using namespace metal;

float mul(float a, float b) {
	return a * b;
}
";

#[cfg(target_os = "linux")]
const PRELUDE: &str = "
#include \"cuda_fp16.h\"
__device__ float mul(float a, float b) {
	return a * b;
}
";

fn stack(
    inputs: &[Input],
    instruction: impl ToString,
    reduce: Option<usize>,
    frames: &[(usize, &[usize], usize)],
) -> Stack {
    Stack {
        inputs: inputs.to_vec(),
        instruction: instruction.to_string(),
        frames: frames
            .iter()
            .enumerate()
            .map(|(i, (size, strides, out))| StackFrame {
                size: *size,
                input_strides: strides.to_vec(),
                output_stride: *out,
                reduce: reduce.map(|n| n == i).unwrap_or_default(),
                ..Default::default()
            })
            .collect(),
        kernel_output: false,
    }
}

fn naive_matmul() {
    // Naive matmul
    let kernels = create_kernels(vec![
        stack(
            &[Input::Inp(0), Input::Inp(1)],
            "mul",
            None,
            &[(8, &[8, 0], 8), (8, &[0, 1], 1), (8, &[1, 8], 1)],
        ),
        stack(
            &[Input::Ref(0)],
            "sum",
            Some(2),
            &[(8, &[8], 8), (8, &[1], 1), (8, &[1], 0)],
        ),
    ]);

    println!("Tiled Matmul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        shared_buffers,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
        println!("shared: {:?}", shared_buffers);
    }
    println!("---");

    // Run
    let a = (0..64).map(|i| i as f32).collect_vec();
    let b = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect_vec();
    let c = run_graph(vec![a.clone(), b], &kernels).pop().unwrap();
    println!("Out: {c:?}");
    assert_eq!(a, c);
}

fn tiled_matmul() {
    // This is a 8x8x8 tiled matmul. Currently we are just doing tiled loop structure but not loading a tile into smem.
    // We need to detect when we can.
    let kernels = create_kernels(vec![
        stack(
            &[Input::Inp(0), Input::Inp(1)],
            "mul",
            None,
            &[
                (4, &[16, 0], 16),
                (4, &[0, 2], 2),
                (1, &[0, 0], 1),
                (4, &[2, 16], 4),
                (2, &[8, 0], 2),
                (2, &[0, 1], 1),
                (2, &[1, 8], 1),
            ],
        ),
        stack(
            &[Input::Ref(0)],
            "sum",
            Some(6),
            &[
                (4, &[16], 16),
                (4, &[2], 2),
                (1, &[1], 1),
                (4, &[4], 4),
                (2, &[2], 2),
                (2, &[1], 1),
                (2, &[1], 1),
            ],
        ),
        stack(
            &[Input::Ref(1)],
            "sum",
            Some(5),
            &[
                (4, &[16], 16),
                (4, &[2], 2),
                (1, &[1], 1),
                (2, &[2], 8),
                (2, &[1], 1),
                (4, &[4], 4),
            ],
        ),
    ]);

    println!("Tiled Matmul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        shared_buffers,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
        println!("shared: {:?}", shared_buffers);
    }
    println!("---");

    // Run
    let a = (0..64).map(|i| i as f32).collect_vec();
    let b = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect_vec();
    let c = run_graph(vec![a.clone(), b], &kernels).pop().unwrap();
    println!("Out: {c:?}");
    assert_eq!(a, c)
}

fn tiled_matmul_smem() {
    // This is a 8x8x8 tiled matmul. Loads tiles into smem
    let kernels = create_kernels(vec![
        stack(
            &[Input::Inp(0)],
            "smem_load",
            None,
            &[
                (4, &[16], 16),
                (4, &[0], 2),
                (1, &[0], 0),
                (4, &[2], 4),
                (2, &[8], 2),
                (2, &[1], 1),
            ],
        ),
        stack(
            &[Input::Inp(1)],
            "smem_load",
            None,
            &[
                (4, &[0], 16),
                (4, &[2], 2),
                (1, &[0], 0),
                (4, &[16], 4),
                (2, &[8], 2),
                (2, &[1], 1),
            ],
        ),
        stack(
            &[Input::Ref(0), Input::Ref(1)],
            "mul",
            None,
            &[
                (4, &[16, 16], 16),
                (4, &[2, 2], 2),
                (1, &[0, 0], 0),
                (4, &[2, 16], 2),
                (2, &[0, 1], 1),
                (2, &[2, 0], 8),
                (2, &[1, 2], 1),
            ],
        ),
        stack(
            &[Input::Ref(2)],
            "sum",
            Some(6),
            &[
                (4, &[16], 16),
                (4, &[2], 2),
                (1, &[0], 0),
                (4, &[2], 1),
                (2, &[1], 1),
                (2, &[8], 8),
                (2, &[1], 1),
            ],
        ),
        stack(
            &[Input::Ref(3)],
            "sum",
            Some(5),
            &[
                (4, &[16], 16),
                (4, &[2], 2),
                (1, &[0], 0),
                (2, &[1], 1),
                (2, &[8], 8),
                (4, &[1], 1),
            ],
        ),
    ]);

    println!("Tiled Matmul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        shared_buffers,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
        println!("shared: {:?}", shared_buffers);
    }
    println!("---");

    // Run
    let a = (0..64).map(|i| i as f32).collect_vec();
    let b = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect_vec();
    let c = run_graph(vec![a.clone(), b], &kernels).pop().unwrap();
    println!("Out: {c:?}");
    assert_eq!(a, c)
}

fn tiled_matmul_simdgroup() {
    // This is a 8x8x8 tiled matmul. Uses simgroup
    let kernels = create_kernels(vec![
        stack(
            &[Input::Inp(0)],
            "simdgroup_load",
            None,
            &[
                (512, &[32768], 32768),
                (512, &[0], 8),
                (1, &[0], 0),
                (8, &[0], 4096),
                (4, &[0], 2),
                (1, &[0], 0),
                (512, &[8], 8),
            ],
        ),
        stack(
            &[Input::Inp(1)],
            "simdgroup_load",
            None,
            &[
                (512, &[0], 32768),
                (512, &[8], 8),
                (1, &[0], 0),
                (8, &[0], 4096),
                (4, &[0], 2),
                (1, &[0], 0),
                (512, &[32768], 8),
            ],
        ),
        stack(
            &[Input::Ref(0), Input::Ref(1)],
            "simdgroup_multiply_accumulate",
            Some(6),
            &[
                (512, &[32768, 32768], 32768),
                (512, &[8, 8], 8),
                (1, &[0, 0], 0),
                (8, &[4096, 4096], 0),
                (4, &[2, 2], 0),
                (1, &[0, 0], 0),
                (512, &[8, 8], 1),
            ],
        ),
    ]);

    println!("Tiled Simdgroup Matmul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        shared_buffers,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
        println!("shared: {:?}", shared_buffers);
    }
    println!("---");

    // Run
    let a = (0..4096).map(|i| i as f32).collect_vec();
    let b = (0..4096)
        .flat_map(|i| (0..4096).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect_vec();
    let c = run_graph(vec![a.clone(), b.clone()], &kernels)
        .pop()
        .unwrap();
    for (ind, (i, j)) in a.into_iter().zip(c).enumerate() {
        if (i - j).abs() > 1e-3 {
            panic!("Mismatch at index {ind}: {i} | {j}");
        }
    }
}

fn tiled_matmul_simdgroup_vector() {
    // This is a 8x8x8 tiled matmul. Uses simgroup
    let kernels = create_kernels(vec![
        stack(
            &[Input::Inp(0)],
            "simdgroup_load",
            None,
            &[
                (128, &[131072], 131072),
                (128, &[0], 32),
                (1, &[0], 0),
                (32, &[0], 4096),
                (1, &[0], 2),
                (1, &[0], 0),
                (4, &[0], 32768),
                (4, &[32768], 1),
                (512, &[8], 8),
            ],
        ),
        stack(
            &[Input::Inp(1)],
            "simdgroup_load",
            None,
            &[
                (128, &[0], 131072),
                (128, &[32], 32),
                (1, &[0], 0),
                (32, &[0], 4096),
                (1, &[0], 2),
                (1, &[0], 0),
                (4, &[8], 1),
                (4, &[0], 32768),
                (512, &[32768], 8),
            ],
        ),
        stack(
            &[Input::Ref(0), Input::Ref(1)],
            "simdgroup_multiply_accumulate",
            Some(8),
            &[
                (128, &[131072, 131072], 131072),
                (128, &[32, 32], 32),
                (1, &[0, 0], 1),
                (32, &[4096, 4096], 1),
                (1, &[2, 2], 1),
                (1, &[0, 0], 1),
                (4, &[32768, 1], 1),
                (4, &[1, 32768], 1),
                (512, &[8, 8], 1),
            ],
        ),
        stack(
            &[Input::Ref(2)],
            "simdgroup_store",
            None,
            &[
                (128, &[131072], 131072),
                (128, &[32], 32),
                (1, &[1], 0),
                (32, &[1], 0),
                (1, &[1], 0),
                (1, &[1], 0),
                (4, &[1], 8),
                (4, &[1], 32768),
            ],
        ),
    ]);

    println!("Tiled Simdgroup Matmul 4x1");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        shared_buffers,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
        println!("shared: {:?}", shared_buffers);
    }
    println!("---");

    // Run
    let a = (0..4096).map(|i| i as f32).collect_vec();
    let b = (0..4096)
        .flat_map(|i| (0..4096).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect_vec();
    let c = run_graph(vec![a.clone(), b.clone()], &kernels)
        .pop()
        .unwrap();
    for (ind, (i, j)) in a.into_iter().zip(c).enumerate() {
        if (i - j).abs() > 1e-3 {
            panic!("Mismatch at index {ind}: {i} | {j}");
        }
    }
}

fn exp_sin_outer_product() {
    // This pulls in a batch of 3 vectors of 4, takes exp and sin of them, and then does an outer product of those vectors
    // Currently it needs two kernels to do this. Should be possible to merge them into one and use shared mem to store the intermediates and a threadblock barrier
    let kernels = create_kernels(vec![
        Stack {
            inputs: vec![Input::Inp(0)],
            instruction: "exp".to_string(),
            frames: vec![
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 3,
                    input_strides: vec![4],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![1],
                    output_stride: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        Stack {
            inputs: vec![Input::Inp(0)],
            instruction: "sin".to_string(),
            frames: vec![
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 3,
                    input_strides: vec![4],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![1],
                    output_stride: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        Stack {
            inputs: vec![Input::Ref(0), Input::Ref(1)],
            instruction: "mul".to_string(),
            frames: vec![
                StackFrame {
                    size: 1,
                    input_strides: vec![0, 0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0, 0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0, 0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 3,
                    input_strides: vec![4, 4],
                    output_stride: 16,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![1, 0],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![0, 1],
                    output_stride: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    ]);

    println!("Exp-Sin Outer Product");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        ..
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
    }
    println!("---");
    let a = (0..12).map(|i| i as f32).collect_vec();
    println!("Out: {:?}", run_graph(vec![a], &kernels).last().unwrap());
}

fn shared_exp() {
    // This does Tensor(3, 4).mul(Tensor(3).exp().expand(4, dim=1)).sum_reduce(dim=1)
    let kernels = create_kernels(vec![
        Stack {
            inputs: vec![Input::Inp(1)],
            instruction: "exp".to_string(),
            frames: vec![StackFrame {
                size: 3,
                input_strides: vec![1],
                output_stride: 1,
                ..Default::default()
            }],
            ..Default::default()
        },
        Stack {
            inputs: vec![Input::Inp(0), Input::Ref(0)],
            instruction: "mul".to_string(),
            frames: vec![
                StackFrame {
                    size: 3,
                    input_strides: vec![4, 1],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![1, 0],
                    output_stride: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        Stack {
            inputs: vec![Input::Ref(1)],
            instruction: "sum".to_string(),
            frames: vec![
                StackFrame {
                    size: 3,
                    input_strides: vec![4],
                    output_stride: 1,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![1],
                    output_stride: 0,
                    reduce: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    ]);

    println!("Shared exp vector mul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
        ..
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
    }
    println!("---");

    // Run
    let matrix = (0..12).map(|i| i as f32).collect::<Vec<_>>();
    let vector = vec![1.0, 2.0, 3.0];
    let c = run_graph(vec![vector, matrix], &kernels).pop().unwrap();
    for (a, b) in [16.30969, 162.55922, 763.2503].iter().zip(c) {
        if (a - b).abs() > 1e-3 {
            panic!("{a} != {b}");
        }
    }
}

fn main() {
    // shared_exp();
    tiled_matmul_simdgroup_vector();
}

// Validate some properties about the graph
fn validate_graph(graph: &[Stack]) {
    let mut used_inputs = vec![];
    for (n_stack, stack) in graph.iter().enumerate() {
        for input in &stack.inputs {
            match input {
                Input::Inp(i) => {
                    if *i >= used_inputs.len() {
                        used_inputs.extend((0..(i - used_inputs.len() + 1)).map(|_| false));
                        used_inputs[*i] = true;
                    } else {
                        used_inputs[*i] = true;
                    }
                }
                Input::Ref(i) => {
                    assert!(
                        *i < n_stack,
                        "Stack trying to use an input from a stack that hasn't been ran yet!"
                    );
                }
            }
        }
        for frame in &stack.frames {
            assert_eq!(
                stack.inputs.len(),
                frame.input_strides.len(),
                "Number of frame input strides don't match number of stack inputs!"
            );
        }
    }

    assert!(
        used_inputs.iter().all(|i| *i),
        "Not all input buffers are used!"
    );
}

#[cfg(target_os = "linux")]
fn run_graph(inputs: Vec<Vec<f32>>, kernels: &[Kernel]) -> Vec<Vec<f32>> {
    use cudarc::{
        driver::{CudaContext, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Allocate input buffers
    let mut buffers = inputs
        .iter()
        .map(|buf| {
            let mut a = stream.alloc_zeros::<f32>(buf.len()).unwrap();
            stream.memcpy_htod(buf, &mut a).unwrap();
            a
        })
        .collect_vec();
    let n_orig_buffers = buffers.len();
    // Allocate output buffers
    for kernel in kernels {
        for output in &kernel.outputs {
            buffers.push(stream.alloc_zeros::<f32>(*output).unwrap());
        }
    }
    // Queue up kernels
    let mut output_kernel_index = 0;
    for (
        n_kernel,
        Kernel {
            code,
            grid,
            threadblock,
            inputs,
            outputs,
            shared_buffers,
        },
    ) in kernels.iter().enumerate()
    {
        // Compile kernel
        let ptx = compile_ptx(code).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = module.load_function(&format!("kernel{n_kernel}")).unwrap();
        let mut launch_args = stream.launch_builder(&f);
        let (input_buffers, output_buffers) =
            buffers.split_at_mut(output_kernel_index + n_orig_buffers);
        // Set inputs
        for input in inputs {
            launch_args.arg(&input_buffers[*input]);
        }
        // Set outputs
        for n in output_buffers.iter_mut().take(outputs.len()) {
            launch_args.arg(n);
        }
        output_kernel_index += outputs.len();
        // // Set shared buffers
        // for (i, buf) in shared_buffers.iter().enumerate() {
        //     encoder
        //         .set_threadgroup_memory_length(i as u64, (buf * std::mem::size_of::<f32>()) as u64);
        // }

        // Set dispatch
        let cfg = LaunchConfig {
            grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
            block_dim: (
                threadblock.0 as u32,
                threadblock.1 as u32,
                threadblock.2 as u32,
            ),
            shared_mem_bytes: shared_buffers.iter().sum::<usize>() as u32,
        };

        // Run
        unsafe { launch_args.launch(cfg) }.unwrap();
    }

    // Copy back intermediate and output buffers
    let mut data = vec![];
    for buffer in &buffers[inputs.len()..] {
        data.push(stream.memcpy_dtov(buffer).unwrap());
    }
    data
}

#[cfg(target_os = "macos")]
fn run_graph(inputs: Vec<Vec<f32>>, kernels: &[Kernel]) -> Vec<Vec<f32>> {
    use metal_rs::{
        CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device,
        MTLResourceOptions, MTLSize,
    };
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();

    // Allocate input buffers
    let mut buffers = inputs
        .iter()
        .map(|buf| {
            device.new_buffer_with_data(
                buf.as_ptr() as *mut _,
                (buf.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect_vec();
    let n_orig_buffers = buffers.len();
    // Allocate output buffers
    for kernel in kernels {
        for output in &kernel.outputs {
            buffers.push(device.new_buffer(
                (output * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
    }
    // Queue up kernels
    let mut output_kernel_index = 0;
    for (
        n_kernel,
        Kernel {
            code,
            grid,
            threadblock,
            inputs,
            outputs,
            shared_buffers,
        },
    ) in kernels.iter().enumerate()
    {
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

        // Compile kernel
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        let lib = device.new_library_with_source(code, &options).unwrap();
        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(
            &lib.get_function(&format!("kernel{n_kernel}"), None)
                .unwrap(),
        ));
        let pipeline = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();
        encoder.set_compute_pipeline_state(&pipeline);

        // Set inputs
        for (i, input) in inputs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(&buffers[*input]), 0);
        }
        // Set output
        for n in 0..outputs.len() {
            encoder.set_buffer(
                (inputs.len() + n) as u64,
                Some(&buffers[output_kernel_index + n_orig_buffers]),
                0,
            );
            output_kernel_index += 1;
        }
        // Set shared buffers
        for (i, buf) in shared_buffers.iter().enumerate() {
            encoder
                .set_threadgroup_memory_length(i as u64, (buf * std::mem::size_of::<f32>()) as u64);
        }

        // Set dispatch
        encoder.dispatch_thread_groups(
            MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
            MTLSize::new(
                threadblock.0 as u64,
                threadblock.1 as u64,
                threadblock.2 as u64,
            ),
        );
        encoder.end_encoding();
    }

    // Run
    let start = std::time::Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    println!("Ran in {}ms", start.elapsed().as_millis());

    // Copy back intermediate and output buffers
    let mut data = vec![];
    for buffer in &buffers[inputs.len()..] {
        let mut curr_data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = buffer.contents() as *mut f32;
        for (i, d) in curr_data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        data.push(curr_data)
    }
    data
}

fn create_kernels(mut ir: Vec<Stack>) -> Vec<Kernel> {
    validate_graph(&ir);
    ir.last_mut().unwrap().kernel_output = true;
    // Merge the stacks as much as possible
    let mut loop_dim = 0;
    let mut merged_ir = ir
        .iter()
        .cloned()
        .map(|mut v| {
            if v.instruction == "sum" || v.instruction == "simdgroup_multiply_accumulate" {
                // Assign loop char
                if let Some(frame) = v.frames.iter_mut().find(|f| f.reduce) {
                    frame.loop_char = Some((b'a' + (loop_dim % 26) as u8) as char);
                }
                loop_dim += 1;
            }
            let outer = v.clone();
            for (i, inp) in v.inputs.iter_mut().enumerate() {
                *inp = Input::Inp(i);
            }
            (outer, vec![v])
        })
        .collect_vec();
    // Map from logical stack indexes to (kernel index, inside kernel stack index)
    let mut logical_to_physical_stack_map = (0..ir.len())
        .map(|i| (i, (i, 0)))
        .collect::<HashMap<_, _>>();
    let mut no_match = false;
    while !no_match {
        no_match = true;
        for l in 0..(merged_ir.len() - 1) {
            println!(
                "-------------------\n{}",
                merged_ir
                    .iter()
                    .map(|(a, i)| format!(
                        "{:?}\n{}",
                        a.inputs,
                        i.iter().map(|s| s.to_string()).join("\n"),
                    ))
                    .join("\n---\n")
            );
            // Try to match ir[l] and ir[l + 1]
            let check_match = |a: &[StackFrame],
                               b: &[StackFrame],
                               real_a: &[Stack],
                               dep_inputs: &Vec<(usize, usize)>,
                               a_inst: &str| {
                if dep_inputs.is_empty() {
                    a.iter().filter(|l| !l.reduce).count() == b.len()
                } else {
                    // Check strides for each input
                    dep_inputs.iter().all(|(inp, reference)| {
                        println!("Inp : {inp}");
                        // Filter for non-reduced frames
                        let a_frames = a
                            .iter()
                            .cloned()
                            .zip(&real_a[*reference].frames)
                            .map(|(mut l, real)| {
                                l.output_stride = real.output_stride;
                                l
                            })
                            .filter(|l| !l.reduce)
                            .collect_vec();
                        // Filter for only input frames that matter (they don't if the stride is zero on a non-filler frame)
                        let b_frames = b
                            .iter()
                            .filter(|l| l.size == 0 || l.size == 1 || l.input_strides[*inp] != 0)
                            .collect_vec();
                        println!("A: {}", a_frames.len());
                        println!("A: {:?}", a_frames);
                        println!("B: {}", b_frames.len());
                        a_frames.len() == b_frames.len()
                            && a_frames
                                .iter()
                                .zip(b_frames)
                                .enumerate()
                                .all(|(n_frame, (a, b))| {
                                    println!(
                                        "{} | {} and {} | {} ({})",
                                        a.size,
                                        b.size,
                                        a.output_stride,
                                        b.input_strides[*inp],
                                        a.input_strides[0]
                                    );
                                    a.size == b.size
                                        && (a.output_stride == b.input_strides[*inp]
                                            	// It is ok if we have mismatched output - input strides if this is going through shared memory in a threadblock
                                                || n_frame >= 3 && a_inst == "smem_load")
                                })
                    })
                }
            };
            let dep_inputs = merged_ir[l + 1]
                .0
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(ind, inp)| {
                    if let Input::Ref(i) = inp {
                        if logical_to_physical_stack_map[i].0 == l {
                            Some((ind, i - l))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect_vec();
            let mut matched = check_match(
                &merged_ir[l].0.frames,
                &merged_ir[l + 1].0.frames,
                &merged_ir[l].1,
                &dep_inputs,
                &merged_ir[l].0.instruction,
            );
            let mut orig_reduce_dim = None;
            if !matched {
                if let Some(orig_dim) = merged_ir[l + 1].0.frames.iter().position(|i| i.reduce) {
                    // Try to slide the reduce until we get a match
                    let mut dim = orig_dim;
                    for i in 0..merged_ir[l + 1].0.frames.len() {
                        let e = merged_ir[l + 1].0.frames.remove(dim);
                        merged_ir[l + 1].0.frames.insert(i, e);
                        if check_match(
                            &merged_ir[l].0.frames,
                            &merged_ir[l + 1].0.frames,
                            &merged_ir[l].1,
                            &dep_inputs,
                            &merged_ir[l].0.instruction,
                        ) {
                            // Found a match!
                            matched = true;
                            // Record original dim
                            orig_reduce_dim = Some(orig_dim);
                            break;
                        }
                        dim = i;
                    }
                    if !matched {
                        // No match, restore to orig dim
                        let e = merged_ir[l + 1].0.frames.remove(dim);
                        merged_ir[l + 1].0.frames.insert(orig_dim, e);
                    }
                }
            }
            if matched {
                // Merge l and l + 1
                for (b, a) in merged_ir[l]
                    .0
                    .frames
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.reduce)
                    .map(|(i, _)| i)
                    .enumerate()
                    .collect_vec()
                {
                    // Set output strides
                    merged_ir[l].0.frames[a].output_stride =
                        merged_ir[l + 1].0.frames[b].output_stride;
                    // Set reduction dims
                    merged_ir[l].0.frames[a].reduce = merged_ir[l + 1].0.frames[b].reduce;
                }
                if merged_ir[l + 1].0.instruction == "sum"
                    || merged_ir[l + 1].0.instruction == "simdgroup_multiply_accumulate"
                {
                    // Need to set reduce dim iterator name
                    let reduce_dim = merged_ir[l + 1]
                        .0
                        .frames
                        .iter()
                        .position(|s| s.reduce)
                        .unwrap();
                    println!("reduce: {reduce_dim}");
                    let loop_char = if let Some(s) = orig_reduce_dim {
                        // Need to correct for the sliding done during matching
                        merged_ir[l + 1].1[0].frames[s].loop_char
                    } else {
                        merged_ir[l + 1].1[0].frames[reduce_dim].loop_char
                    };
                    for Stack { frames, .. } in merged_ir[l].1.iter_mut().rev() {
                        if frames.len() <= reduce_dim {
                            break;
                        }
                        println!("Setting {reduce_dim}");
                        frames
                            .iter_mut()
                            .filter(|s| !s.reduce)
                            .nth(reduce_dim)
                            .unwrap()
                            .loop_char = loop_char;
                        println!("{:?}", frames);
                    }
                }
                // Merge inputs
                // Update logical to physical map
                let first_kernel_size = merged_ir[l].1.len();
                let second_kernel_size = merged_ir[l + 1].1.len();
                let logical_start = merged_ir.iter().take(l).map(|i| i.1.len()).sum::<usize>();
                let logical_end = logical_start + first_kernel_size + second_kernel_size;
                for (i, log) in (logical_start + first_kernel_size..logical_end).enumerate() {
                    logical_to_physical_stack_map.insert(log, (l, i + first_kernel_size));
                }
                // Remap inner inputs
                let first_stack_inputs = merged_ir[l].0.inputs.len();
                let mut inputs_to_push = vec![];
                let actual_inputs = merged_ir[l + 1].0.inputs.clone();
                for stack in &mut merged_ir[l + 1].1 {
                    for (n_inp, input) in stack.inputs.iter_mut().enumerate() {
                        if let Input::Inp(i) = *input {
                            let actual_inp = actual_inputs[i];
                            if let Input::Ref(i) = actual_inp {
                                let (kernel, st) = logical_to_physical_stack_map[&i];
                                if kernel == l {
                                    *input = Input::Ref(st);
                                } else {
                                    inputs_to_push.push((n_inp, Input::Ref(i)));
                                    *input =
                                        Input::Inp(first_stack_inputs + inputs_to_push.len() - 1);
                                }
                            } else if let Input::Inp(i) = actual_inp {
                                inputs_to_push.push((n_inp, Input::Inp(i)));
                                *input = Input::Inp(first_stack_inputs + inputs_to_push.len() - 1);
                            }
                        } else if let Input::Ref(i) = input {
                            *i += first_kernel_size;
                        }
                    }
                }
                // Extend frames if needed
                let n_pre_inputs = merged_ir[l].0.inputs.len();
                for frame in merged_ir[l + 1]
                    .0
                    .frames
                    .iter()
                    .skip(merged_ir[l].0.frames.len())
                    .cloned()
                    .collect_vec()
                {
                    merged_ir[l].0.frames.push(StackFrame {
                        input_strides: vec![0; n_pre_inputs],
                        ..frame
                    });
                }
                for (i, inp) in inputs_to_push {
                    merged_ir[l].0.inputs.push(inp);
                    let strides = merged_ir[l + 1]
                        .0
                        .frames
                        .iter()
                        .map(|f| f.input_strides[i])
                        .collect_vec();
                    for (i, frame) in merged_ir[l].0.frames.iter_mut().enumerate() {
                        if i < strides.len() {
                            frame.input_strides.push(strides[i]);
                        } else {
                            frame.input_strides.push(0);
                        }
                    }
                }

                let mut t = merged_ir.remove(l + 1).1;
                merged_ir[l].1.append(&mut t);
                no_match = false;
                break;
            }
        }
    }

    // Fix up thread-level looping
    // (look for frames outside the grid heirarchy and mark them as a loop dim)
    for (_, stacks) in &mut merged_ir {
        for stack in stacks {
            for (i, frame) in stack
                .frames
                .iter_mut()
                .filter(|f| f.loop_char.is_none())
                .skip(6)
                .enumerate()
            {
                frame.loop_char = Some((b'a' + ((loop_dim + i) % 26) as u8) as char);
                frame.thread_loop = true;
            }
        }
    }
    println!(
        "-------------------\n{}",
        merged_ir
            .iter()
            .map(|(a, i)| format!(
                "{:?}\n{}",
                a.inputs,
                i.iter().map(|s| s.to_string()).join("\n"),
            ))
            .join("\n---\n")
    );

    println!("Final");
    let mut kernels = vec![];
    // Mark cross kernel buffers as outputs
    for (outer, _) in merged_ir.clone() {
        // Change inputs if they reference anything outside this kernel
        for inp in outer.inputs {
            if let Input::Ref(i) = inp {
                let (kernel, stack) = logical_to_physical_stack_map[&i];
                merged_ir[kernel].1[stack].kernel_output = true;
            }
        }
    }

    // Write kernels
    let mut loop_levels: Vec<char> = vec![];
    for (n_kernel, (stack, instructions)) in merged_ir.into_iter().enumerate() {
        // Compute grid and threadblock dim assignments
        let exec_dims = stack
            .frames
            .iter()
            .filter(|i| !i.reduce)
            .map(|d| d.size.max(1))
            .collect_vec(); // Dims to actually execute (none reduce dims)
        let grid = exec_dims
            .iter()
            .take(3)
            .copied()
            .chain(vec![1; 3_usize.saturating_sub(exec_dims.len())])
            .collect_vec();
        let threadblock = exec_dims
            .iter()
            .skip(3)
            .copied()
            .chain(vec![
                1;
                3_usize
                    .saturating_sub(exec_dims.len().saturating_sub(3))
            ])
            .collect_vec();

        // Get input buffer indexes
        let input_buffer_indexes = instructions
            .iter()
            .flat_map(|i| i.inputs.iter())
            .filter_map(|i| {
                if let Input::Inp(i) = i {
                    Some(*i)
                } else {
                    None
                }
            })
            .sorted()
            .dedup()
            .collect_vec();
        let output_buffer_sizes = instructions
            .iter()
            .filter(|s| s.kernel_output)
            .map(|stack| {
                let out = stack
                    .frames
                    .iter()
                    .filter(|i| i.size != 0 && !i.reduce)
                    .map(|i| i.size)
                    .product::<usize>();
                if stack.instruction.contains("simdgroup") {
                    out * 2 // 2 elements per thread outputted for simdgroup matrix
                } else {
                    out
                }
            })
            .collect_vec();

        // Write kernel
        let threadgroup_buffers = instructions
            .iter()
            .enumerate()
            .filter(|(_, i)| i.instruction == "smem_load")
            .map(|(i, s)| {
                (
                    (b'a' + (i % 26) as u8) as char,
                    s.frames.clone(),
                    s.frames
                        .iter()
                        .filter(|f| f.loop_char.is_none() && f.size != 0)
                        .skip(3) // Skip the grid dims
                        .map(|f| f.size)
                        .product::<usize>(),
                )
            })
            .collect_vec();
        let mut kernel_lines: Vec<(String, Vec<char>)> = vec![];
        let mut kernel_output_num = 0;
        for (
            var_names,
            Stack {
                inputs,
                instruction,
                frames,
                kernel_output,
            },
        ) in instructions.iter().enumerate()
        {
            println!("{instruction}: {:?}", frames);
            let var_name = (b'a' + (var_names % 26) as u8) as char;
            let inputs = inputs
                .iter()
                .zip(get_inputs(
                    frames,
                    false,
                    &inputs
                        .iter()
                        .map(|i| {
                            if let Input::Ref(i) = i {
                                threadgroup_buffers
                                    .iter()
                                    .find(|(c, _, _)| *c == (b'a' + (*i % 26) as u8) as char)
                                    .map(|(_, c, _)| c.clone())
                            } else {
                                None
                            }
                        })
                        .collect_vec(),
                    0,
                ))
                .map(|(inp, index)| match inp {
                    Input::Inp(i) => format!(
                        "{}[{}]",
                        (b'A' + (i % 26) as u8) as char,
                        if index.is_empty() {
                            "0".to_string()
                        } else {
                            index
                        }
                    ),
                    Input::Ref(i) => format!(
                        "{}{}{}",
                        if threadgroup_buffers
                            .iter()
                            .any(|(v, _, _)| *v == (b'a' + (i % 26) as u8) as char)
                        {
                            "shared_"
                        } else {
                            ""
                        },
                        (b'a' + (i % 26) as u8) as char,
                        if threadgroup_buffers
                            .iter()
                            .any(|(v, _, _)| *v == (b'a' + (i % 26) as u8) as char)
                        {
                            if index.is_empty() {
                                "[0]".to_string()
                            } else {
                                format!("[{index}]")
                            }
                        } else {
                            "".to_string()
                        }
                    ),
                })
                .join(", ");

            // Write instruction
            #[cfg(target_os = "linux")]
            const BARRIER: &str = "__syncthreads();";
            #[cfg(target_os = "macos")]
            const BARRIER: &str = "threadgroup_barrier(mem_flags::mem_threadgroup);";

            // Write thread-level loops
            let loop_chars: Vec<char> = frames.iter().filter_map(|f| f.loop_char).collect();

            // End loops
            if let Some((i, _)) = loop_levels
                .iter()
                .enumerate()
                .find(|(i, c)| *i >= loop_chars.len() || loop_chars[*i] != **c)
            {
                println!("ENDING :{i}");
                for (i, loop_char) in loop_levels
                    .iter()
                    .cloned()
                    .enumerate()
                    .skip(i)
                    .collect_vec()
                    .into_iter()
                    .rev()
                {
                    kernel_lines.push(("}".to_string(), loop_levels[..i].to_vec()));
                    loop_levels.pop();
                }
            }

            // Create loops
            if let Some((i, _)) = frames
                .iter()
                .filter_map(|f| f.loop_char)
                .enumerate()
                .find(|(i, c)| *i >= loop_levels.len() || loop_levels[*i] != *c)
            {
                for (
                    i,
                    StackFrame {
                        size, loop_char, ..
                    },
                ) in frames
                    .iter()
                    .filter(|f| f.loop_char.is_some())
                    .enumerate()
                    .skip(i)
                    .collect_vec()
                {
                    let loop_char = loop_char.unwrap();

                    if *size <= 8 {
                        // Only unroll loops <= 8
                        kernel_lines.push((
                            format!("#pragma unroll({size})"),
                            loop_chars[..i + 1].to_vec(),
                        ));
                    }
                    kernel_lines.push((format!("for (int loop_{loop_char} = 0; loop_{loop_char} < {size}; ++loop_{loop_char}) {{"), loop_chars[..i + 1].to_vec()));
                    loop_levels.push(loop_char);
                }
            }

            if instruction == "sum" {
                let StackFrame {
                    size, loop_char, ..
                } = frames.iter().find(|f| f.reduce).unwrap();
                let loop_char = loop_char.unwrap();
                let loop_chars: Vec<char> = frames
                    .iter()
                    .filter_map(|f| if f.reduce { None } else { f.loop_char })
                    .collect();
                let mut start_loop_ind = kernel_lines
                    .iter()
                    .position(|f| f.1.contains(&loop_char))
                    .unwrap();
                kernel_lines.insert(
                    start_loop_ind,
                    (format!("float {var_name} = 0.0;"), loop_chars.clone()),
                );
                if *size <= 8 {
                    // Only unroll loops <= 8
                    kernel_lines.insert(
                        start_loop_ind + 1,
                        (format!("#pragma unroll({size})"), loop_chars.clone()),
                    );
                    start_loop_ind += 1;
                }
                kernel_lines.insert(start_loop_ind + 1, (format!("for (int loop_{loop_char} = 0; loop_{loop_char} < {size}; ++loop_{loop_char}) {{"), loop_chars.clone()));
                kernel_lines.push((format!("\t{var_name} += {inputs};"), loop_chars.clone()));
                kernel_lines.push(("}".to_string(), loop_chars));
            } else if instruction == "simdgroup_multiply_accumulate" {
                let StackFrame { loop_char, .. } = frames.iter().find(|f| f.reduce).unwrap();
                let loop_char = loop_char.unwrap();
                let loop_chars: Vec<char> = frames.iter().filter_map(|f| f.loop_char).collect();
                println!("loop char: {loop_char}");
                let start_loop_ind = kernel_lines
                    .iter()
                    .position(|f| f.1.contains(&loop_char))
                    .unwrap();
                println!("pos: {:?}", start_loop_ind);
                kernel_lines.insert(
                    start_loop_ind,
                    (
                        format!("simdgroup_float8x8 {var_name} = simdgroup_float8x8(0);"),
                        frames
                            .iter()
                            .filter(|f| !f.reduce)
                            .filter_map(|f| f.loop_char)
                            .collect(),
                    ),
                );
                kernel_lines.push((
                    format!("simdgroup_multiply_accumulate({var_name}, {inputs}, {var_name});"),
                    loop_chars.clone(),
                ));
            } else if instruction == "simdgroup_store" {
                kernel_lines.push((
                    format!(
                        "simdgroup_store({inputs}, out{kernel_output_num} + {}, 4096);",
                        get_inputs(frames, true, &[], 0)[0]
                    ),
                    frames.iter().filter_map(|f| f.loop_char).collect(),
                ));
            } else if instruction == "smem_load" {
                let loop_chars: Vec<char> = frames.iter().filter_map(|f| f.loop_char).collect();
                if kernel_lines
                    .last()
                    .map(|(l, _)| l == BARRIER)
                    .unwrap_or_default()
                {
                    // If we already have a barrier as the previous instruction, get rid of both.
                    kernel_lines.pop();
                } else {
                    kernel_lines.push((BARRIER.to_string(), loop_chars.clone()));
                }
                kernel_lines.push((
                    format!(
                        "shared_{var_name}[{}] = {inputs};",
                        get_inputs(
                            &frames
                                [3 + kernel_lines.last().map(|i| i.1.len()).unwrap_or_default()..],
                            true,
                            &[],
                            3
                        )[0]
                    ),
                    loop_chars.clone(),
                ));
                kernel_lines.push((BARRIER.to_string(), loop_chars));
            } else if instruction == "simdgroup_load" {
                kernel_lines.push((
                    format!("simdgroup_float8x8 {var_name};"),
                    frames
                        .iter()
                        .filter_map(|f| if f.reduce { None } else { f.loop_char })
                        .collect(),
                ));
                kernel_lines.push((
                    format!(
                        "simdgroup_load({var_name}, {}, 4096);",
                        inputs.replace("[", " + ").replace("]", "")
                    ),
                    frames
                        .iter()
                        .filter_map(|f| if f.reduce { None } else { f.loop_char })
                        .collect(),
                ));
            } else {
                kernel_lines.push((
                    format!("float {var_name} = {instruction}({inputs});"),
                    frames
                        .iter()
                        .filter_map(|f| if f.reduce { None } else { f.loop_char })
                        .collect(),
                ));
            }
            if *kernel_output && !instruction.contains("simdgroup") {
                kernel_lines.push((
                    format!(
                        "out{kernel_output_num}[{}] = {var_name};",
                        get_inputs(frames, true, &[], 0)[0]
                    ),
                    frames
                        .iter()
                        .filter_map(|f| if f.reduce { None } else { f.loop_char })
                        .collect(),
                ));
                kernel_output_num += 1;
            }
        }
        for i in (0..loop_levels.len()).rev() {
            // End loop
            kernel_lines.push(("}".to_string(), loop_levels[..i].to_vec()));
        }
        let mut kernel = kernel_lines
            .into_iter()
            .map(|(s, c)| {
                let n_loops = if s.contains("for (") || s.contains("#pragma") {
                    c.len() - 1
                } else {
                    c.len()
                };
                s.split("\n")
                    .map(|s| format!("{}{s}", (0..n_loops).map(|_| "\t").join("")))
                    .join("\n")
            })
            .join("\n");

        #[allow(unused)]
        let inputs = input_buffer_indexes
            .iter()
            .enumerate()
            .map(|(index, i)| {
                #[cfg(target_os = "macos")]
                {
                    format!(
                        "\tdevice const float* {} [[buffer({index})]]",
                        (b'A' + (i % 26) as u8) as char
                    )
                }
                #[cfg(target_os = "linux")]
                {
                    format!("\tconst float* {}", (b'A' + (i % 26) as u8) as char)
                }
            })
            .chain(
                instructions
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| i.instruction == "smem_load")
                    .enumerate()
                    .map(|(i, (j, _))| {
                        format!(
                            "\tthreadgroup float* shared_{} [[threadgroup({i})]]",
                            (b'a' + (j % 26) as u8) as char
                        )
                    }),
            )
            .join(",\n");
        let outputs = (0..output_buffer_sizes.len())
            .map(|index| {
                #[cfg(target_os = "macos")]
                {
                    format!(
                        "\tdevice float* out{index} [[buffer({})]]",
                        index + input_buffer_indexes.len()
                    )
                }
                #[cfg(target_os = "linux")]
                {
                    format!("\tfloat* out{index}")
                }
            })
            .join(",\n");

        kernel = kernel.split("\n").map(|k| format!("\t{k}")).join("\n");
        #[cfg(target_os = "macos")]
        {
            kernel = format!(
                "{PRELUDE}
kernel void kernel{n_kernel}(
{inputs},
{outputs},
	uint3 blockIdx [[threadgroup_position_in_grid]],
	uint3 threadIdx [[thread_position_in_threadgroup]]
) {{
{kernel}
}}",
            );
        }
        #[cfg(target_os = "linux")]
        {
            kernel = format!(
                "{PRELUDE}
extern \"C\" __global__ void kernel{n_kernel}(
{inputs},
{outputs}
) {{
{kernel}
}}",
            );
        }

        kernels.push(Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            inputs: input_buffer_indexes,
            outputs: output_buffer_sizes,
            shared_buffers: threadgroup_buffers.iter().map(|(_, _, i)| *i).collect(),
        });
    }

    kernels
}

fn get_inputs(
    frames: &[StackFrame],
    output_strides: bool,
    shared_buffer_inputs: &[Option<Vec<StackFrame>>],
    start_dim_index: usize,
) -> Vec<String> {
    // Transpose from indexes[inputs[]] to inputs[indexes[]]
    let mut indexes = vec![
        vec![];
        if output_strides {
            1
        } else {
            frames[0].input_strides.len()
        }
    ];
    for StackFrame {
        input_strides,
        output_stride,
        reduce,
        ..
    } in frames
    {
        if !*reduce {
            if output_strides {
                indexes[0].push(*output_stride);
            } else {
                for (i, s) in input_strides.iter().enumerate() {
                    indexes[i].push(*s);
                }
            }
        }
    }

    indexes
        .iter()
        .enumerate()
        .map(|(n_input, stride)| {
            let mut dim_index = start_dim_index;
            let mut loop_index = 0;
            stride
                .iter()
                .enumerate()
                .flat_map(|(i, s)| {
                    if frames[i].loop_char.is_none() {
                        dim_index += 1;
                        if !output_strides
                            && shared_buffer_inputs[n_input].is_some()
                            && dim_index < 3
                        {
                            return None;
                        }
                    } else {
                        loop_index += 1;
                        if !output_strides
                            && loop_index
                                <= shared_buffer_inputs[n_input]
                                    .as_ref()
                                    .map(|s| s.iter().filter(|f| f.loop_char.is_some()).count())
                                    .unwrap_or_default()
                        {
                            return None;
                        }
                    }
                    if *s == 0 || dim_index > 6 {
                        return None;
                    }
                    Some(match frames[i].loop_char {
                        Some(l) => format!(
                            "loop_{l}{}",
                            if *s == 1 {
                                "".to_string()
                            } else {
                                format!(" * {s}")
                            }
                        ),
                        None => format!(
                            "{}{}",
                            INDEX_TO_DIM[dim_index - 1],
                            if *s == 1 {
                                "".to_string()
                            } else {
                                format!(" * {s}")
                            }
                        ),
                    })
                })
                .join(" + ")
        })
        .collect()
}

const INDEX_TO_DIM: [&str; 6] = [
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "threadIdx.y",
    "threadIdx.z",
];
