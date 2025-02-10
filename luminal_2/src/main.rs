#![allow(clippy::type_complexity)]

// TODO: detect SMEM usage opportunities
// Figure out keeping the exp outside of the loop in the last kernel
// Figure out a general indexing scheme for indexing into non-global buffers (local inputs)
// Run kernels to make sure they work
// Put flattened IR into egglog
// If flattened IR doesn't go into egglog, put nested IR into egglog and write flattening function

use itertools::Itertools;
use metal_rs::{
    CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device, MTLResourceOptions,
    MTLSize,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Input {
    Inp(usize), // An input to this scope
    Ref(usize), // A reference to an earlier variable in the scope
}

const PRELUDE: &str = "
#include <metal_stdlib>
using namespace metal;

float mul(float a, float b) {
	return a * b;
}
";

fn main() {
    // This is a 8x8x8 tiled matmul. Currently we are just doing tiled loop structure but not loading a tile into smem.
    // We need to detect when we can.
    let kernels = create_kernels(vec![
        Stack {
            inputs: vec![Input::Inp(0), Input::Inp(1)],
            instruction: "mul".to_string(),
            frames: vec![
                StackFrame {
                    size: 4,
                    input_strides: vec![16, 0],
                    output_stride: 16,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![0, 2],
                    output_stride: 2,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0, 0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![2, 16],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![8, 0],
                    output_stride: 2,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![0, 1],
                    output_stride: 1,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![1, 8],
                    output_stride: 1,
                    ..Default::default()
                },
            ],
        },
        Stack {
            inputs: vec![Input::Ref(0)],
            instruction: "sum".to_string(),
            frames: vec![
                StackFrame {
                    size: 4,
                    input_strides: vec![16],
                    output_stride: 16,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![2],
                    output_stride: 2,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![4],
                    output_stride: 4,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![2],
                    output_stride: 2,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![1],
                    output_stride: 1,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![1],
                    output_stride: 1,
                    reduce: true,
                    ..Default::default()
                },
            ],
        },
        Stack {
            inputs: vec![Input::Ref(1)],
            instruction: "sum".to_string(),
            frames: vec![
                StackFrame {
                    size: 4,
                    input_strides: vec![16],
                    output_stride: 16,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![2],
                    output_stride: 2,
                    ..Default::default()
                },
                StackFrame {
                    size: 1,
                    input_strides: vec![0],
                    output_stride: 0,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![2],
                    output_stride: 8,
                    ..Default::default()
                },
                StackFrame {
                    size: 2,
                    input_strides: vec![1],
                    output_stride: 1,
                    ..Default::default()
                },
                StackFrame {
                    size: 4,
                    input_strides: vec![4],
                    output_stride: 4,
                    reduce: true,
                    ..Default::default()
                },
            ],
        },
    ]);

    // let kernels = create_kernels(vec![
    //     Stack {
    //         inputs: vec![Input::Inp(0), Input::Inp(1)],
    //         instruction: "mul".to_string(),
    //         frames: vec![
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![8, 0],
    //                 output_stride: 8,
    //                 ..Default::default()
    //             },
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![0, 1],
    //                 output_stride: 1,
    //                 ..Default::default()
    //             },
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![1, 8],
    //                 output_stride: 1,
    //                 ..Default::default()
    //             },
    //         ],
    //     },
    //     Stack {
    //         inputs: vec![Input::Ref(0)],
    //         instruction: "sum".to_string(),
    //         frames: vec![
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![8],
    //                 output_stride: 8,
    //                 ..Default::default()
    //             },
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![1],
    //                 output_stride: 1,
    //                 ..Default::default()
    //             },
    //             StackFrame {
    //                 size: 8,
    //                 input_strides: vec![1],
    //                 output_stride: 0,
    //                 reduce: true,
    //                 ..Default::default()
    //             },
    //         ],
    //     },
    // ]);

    println!("Tiled Matmul");
    for Kernel {
        code,
        grid,
        threadblock,
        inputs,
        outputs,
    } in &kernels
    {
        println!("---");
        println!("Grid: {grid:?} Threadblock: {threadblock:?}");
        println!("{code}");
        println!("inputs: {:?}", inputs);
        println!("outputs: {:?}", outputs);
    }
    println!("---");

    // // This pulls in a batch of 3 vectors of 4, takes exp and sin of them, and then does an outer product of those vectors
    // // Currently it needs two kernels to do this. Should be possible to merge them into one and use shared mem to store the intermediates and a threadblock barrier
    // let kernels = create_kernels(vec![
    //     (
    //         vec![Input::Inp(0)],
    //         "exp".to_string(),
    //         vec![
    //             (3, vec![4], 4, false),
    //             (1, vec![0], 1, false), // Padding dims to push the vector dims to the threadblock
    //             (1, vec![0], 1, false),
    //             (4, vec![1], 1, false),
    //         ],
    //     ),
    //     (
    //         vec![Input::Inp(0)],
    //         "sin".to_string(),
    //         vec![
    //             (3, vec![4], 4, false),
    //             (1, vec![0], 1, false),
    //             (1, vec![0], 1, false),
    //             (4, vec![1], 1, false),
    //         ],
    //     ),
    //     (
    //         vec![Input::Ref(0), Input::Ref(1)],
    //         "mul".to_string(),
    //         vec![
    //             (3, vec![4, 4], 16, false),
    //             (1, vec![0], 1, false),
    //             (1, vec![0], 1, false),
    //             (4, vec![1, 0], 4, false),
    //             (4, vec![1, 0], 4, false),
    //         ],
    //     ),
    // ]);

    // println!("Exp-Sin Outer Product");
    // for Kernel {
    //     code,
    //     grid,
    //     threadblock,
    //     ..
    // } in kernels
    // {
    //     println!("---");
    //     println!("Grid: {grid:?} Threadblock: {threadblock:?}");
    //     println!("{code}");
    // }
    // println!("---");

    // This does Tensor(3, 4).mul(Tensor(3).exp().expand(4, dim=1)).sum_reduce(dim=1)
    // let kernels = create_kernels(vec![
    //     Stack {
    //         inputs: vec![Input::Inp(1)],
    //         instruction: "exp".to_string(),
    //         frames: vec![StackFrame {
    //             size: 3,
    //             input_strides: vec![1],
    //             output_stride: 1,
    //              ..Default::default(),
    //         }],
    //     },
    //     Stack {
    //         inputs: vec![Input::Inp(0), Input::Ref(0)],
    //         instruction: "mul".to_string(),
    //         frames: vec![
    //             StackFrame {
    //                 size: 3,
    //                 input_strides: vec![4, 1],
    //                 output_stride: 4,
    //                  ..Default::default(),
    //             },
    //             StackFrame {
    //                 size: 4,
    //                 input_strides: vec![1, 0],
    //                 output_stride: 1,
    //                  ..Default::default(),
    //             },
    //         ],
    //     },
    //     Stack {
    //         inputs: vec![Input::Ref(1)],
    //         instruction: "sum".to_string(),
    //         frames: vec![
    //             StackFrame {
    //                 size: 3,
    //                 input_strides: vec![4],
    //                 output_stride: 1,
    //                  ..Default::default(),
    //             },
    //             StackFrame {
    //                 size: 4,
    //                 input_strides: vec![1],
    //                 output_stride: 0,
    //                 reduce: true,
    //             },
    //         ],
    //     },
    // ]);

    // println!("Shared exp vector mul");
    // for Kernel {
    //     code,
    //     grid,
    //     threadblock,
    //     inputs,
    //     outputs,
    // } in &kernels
    // {
    //     println!("---");
    //     println!("Grid: {grid:?} Threadblock: {threadblock:?}");
    //     println!("{code}");
    //     println!("inputs: {:?}", inputs);
    //     println!("outputs: {:?}", outputs);
    // }
    // println!("---");

    // Set inputs
    let a = (0..64).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == i { 1.0 } else { 0.0 }))
        .collect::<Vec<_>>();

    println!("Out: {:?}", run_graph(vec![a, b], &kernels));
}

fn run_graph(inputs: Vec<Vec<f32>>, kernels: &[Kernel]) -> Vec<f32> {
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
        .collect::<Vec<_>>();
    let n_orig_buffers = buffers.len();
    // Allocate output buffers
    for kernel in kernels {
        assert_eq!(
            kernel.outputs.len(),
            1,
            "Can't handle more than one kernel output for now"
        );
        buffers.push(device.new_buffer(
            (kernel.outputs[0] * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
    }
    // Queue up kernels
    for (
        n_kernel,
        Kernel {
            code,
            grid,
            threadblock,
            inputs,
            ..
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
        encoder.set_buffer(
            inputs.len() as u64,
            Some(&buffers[n_kernel + n_orig_buffers]),
            0,
        );

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
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy back last buffer
    let buffer = buffers.last().unwrap();
    let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
    let ptr = buffer.contents() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

#[derive(Clone, Debug)]
struct Stack {
    inputs: Vec<Input>,
    instruction: String,
    frames: Vec<StackFrame>,
}

#[derive(Clone, Debug, Default)]
struct StackFrame {
    size: usize,
    input_strides: Vec<usize>,
    loop_char: Option<char>,
    output_stride: usize,
    reduce: bool,
}

#[derive(Debug, Clone)]
struct Kernel {
    code: String,
    grid: (usize, usize, usize),
    threadblock: (usize, usize, usize),
    inputs: Vec<usize>,  // global buffer indexes this kernel uses
    outputs: Vec<usize>, // buffer sizes this kernel creates
}

fn create_kernels(ir: Vec<Stack>) -> Vec<Kernel> {
    // Merge the stacks as much as possible
    let mut loop_dim = 0;
    let mut merged_ir = ir
        .iter()
        .cloned()
        .map(|mut v| {
            if v.instruction == "sum" {
                loop_dim += 1;
            }
            for frame in &mut v.frames {
                frame.loop_char = if v.instruction == "sum" {
                    Some((b'a' + ((loop_dim - 1) % 26) as u8) as char)
                } else {
                    None
                };
            }
            (v.clone(), vec![v])
        })
        .collect::<Vec<_>>();
    let mut no_match = false;
    while !no_match {
        no_match = true;
        let mut logical_index = 0;
        for l in 0..(merged_ir.len() - 1) {
            // Try to match ir[l] and ir[l + 1]
            let check_match = |a: &[StackFrame], b: &[StackFrame], is_dep: bool| {
                a.iter().filter(|l| !l.reduce).count() == b.len()
                    && a.iter().filter(|l| !l.reduce).zip(b.iter()).all(|(a, b)| {
                        a.size == b.size
                            && (!is_dep
                                || (a.output_stride == b.input_strides[0]
                                    && b.input_strides.len() == 1))
                    })
            };
            let dep_inputs = merged_ir[l + 1]
                .0
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, inp)| {
                    if let Input::Ref(i) = inp {
                        *i < logical_index + merged_ir[l].1.len() && *i >= logical_index
                    } else {
                        false
                    }
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let mut matched = check_match(
                &merged_ir[l].0.frames,
                &merged_ir[l + 1].0.frames,
                !dep_inputs.is_empty(),
            );
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
                            !dep_inputs.is_empty(),
                        ) {
                            // Found a match!
                            matched = true;
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
                    .collect::<Vec<_>>()
                {
                    // Set output strides
                    merged_ir[l].0.frames[a].output_stride =
                        merged_ir[l + 1].0.frames[b].output_stride;
                    // Set reduction dims
                    merged_ir[l].0.frames[a].reduce = merged_ir[l + 1].0.frames[b].reduce;
                }
                // Set input strides for incoming kernel
                for dep_input in dep_inputs {
                    // Input dep_input from l + 1 is a dependency input. Let's set it's input stride to 0 for now.
                    // Note this is not always correct! Once we do more complex input sharing we need to index into local variables
                    let input_ref = merged_ir[l + 1].0.inputs[dep_input];
                    for Stack { inputs, frames, .. } in &mut merged_ir[l + 1].1 {
                        if let Some(dep_pos) = inputs.iter().position(|i| *i == input_ref) {
                            for StackFrame { input_strides, .. } in frames {
                                input_strides[dep_pos] = 0;
                            }
                        }
                    }
                }
                if merged_ir[l + 1].0.instruction == "sum" {
                    // Need to set reduce dim iterator name
                    let reduce_dim = merged_ir[l + 1]
                        .0
                        .frames
                        .iter()
                        .position(|s| s.reduce)
                        .unwrap();
                    let loop_char = merged_ir[l + 1].1[0].frames[reduce_dim].loop_char;
                    for Stack { frames, .. } in &mut merged_ir[l].1 {
                        assert!(frames.len() > reduce_dim, "Are we sharing a dim?");
                        frames
                            .iter_mut()
                            .filter(|s| !s.reduce)
                            .nth(reduce_dim)
                            .unwrap()
                            .loop_char = loop_char;
                    }
                }
                let mut t = merged_ir.remove(l + 1).1;
                merged_ir[l].1.append(&mut t);
                no_match = false;
                break;
            }
            logical_index += merged_ir[l].1.len();
        }
    }

    let mut kernels = vec![];
    let mut logical_index_start_kernel = 0;
    let start_internal_buffer_index = merged_ir
        .iter()
        .flat_map(|(_, inst)| inst.iter().flat_map(|i| i.inputs.iter()))
        .filter_map(|i| {
            if let Input::Inp(i) = i {
                Some(i + 1)
            } else {
                None
            }
        })
        .max()
        .unwrap_or_default();
    let logical_indexes_to_kernel_indexes = merged_ir
        .iter()
        .enumerate()
        .flat_map(|(i, inner)| inner.1.iter().map(move |_| i))
        .collect::<Vec<_>>();
    println!("{:?}", merged_ir);
    for (n_kernel, (stack, mut instructions)) in merged_ir.into_iter().enumerate() {
        // Compute grid and threadblock dim assignments
        let exec_dims = stack
            .frames
            .iter()
            .filter(|i| !i.reduce)
            .map(|d| d.size)
            .collect::<Vec<_>>();
        let grid = exec_dims
            .iter()
            .take(3)
            .cloned()
            .chain((0..(3_usize.saturating_sub(exec_dims.len()))).map(|_| 1))
            .collect::<Vec<_>>();
        let threadblock = exec_dims
            .iter()
            .skip(3)
            .cloned()
            .chain((0..(3_usize.saturating_sub(exec_dims.len().saturating_sub(3)))).map(|_| 1))
            .collect::<Vec<_>>();

        // TODO: detect when we can use shared mem

        // Change inputs if they reference anything outside this kernel
        for Stack { inputs, .. } in &mut instructions {
            for inp in inputs {
                if let Input::Ref(i) = *inp {
                    if i < logical_index_start_kernel {
                        *inp = Input::Inp(
                            logical_indexes_to_kernel_indexes[i] + start_internal_buffer_index,
                        );
                    } else {
                        *inp = Input::Ref(i - logical_index_start_kernel);
                    }
                }
            }
        }
        // Get input buffer indexes
        let mut input_buffer_indexes = instructions
            .iter()
            .flat_map(|i| i.inputs.iter())
            .filter_map(|i| {
                if let Input::Inp(i) = i {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        input_buffer_indexes.sort();
        input_buffer_indexes.dedup();
        let output_buffer_size = stack
            .frames
            .iter()
            .filter(|i| !i.reduce)
            .map(|i| i.size)
            .product::<usize>();

        // Write kernels
        let mut kernel = "".to_string();
        let mut var_names = 0;
        for Stack {
            inputs,
            instruction,
            frames,
        } in &instructions
        {
            let var_name = (b'a' + (var_names % 26) as u8) as char;
            var_names += 1;
            let inputs = inputs
                .iter()
                .zip(get_inputs(frames))
                .map(|(inp, index)| {
                    let var = match inp {
                        Input::Inp(i) => (b'A' + (i % 26) as u8) as char,
                        Input::Ref(i) => (b'a' + (i % 26) as u8) as char,
                    };
                    let ind = if index.is_empty() {
                        "".to_string()
                    } else {
                        format!("[{index}]")
                    };
                    format!("{var}{ind}")
                })
                .join(", ");
            if instruction == "sum" {
                let StackFrame {
                    size, loop_char, ..
                } = frames.iter().find(|f| f.reduce).unwrap();
                let loop_char = loop_char.unwrap();
                kernel = format!(
                    "float {var_name} = 0.0;
for (int loop_{loop_char} = 0; loop_{loop_char} < {size}; ++loop_{loop_char}) {{
{}
	{var_name} += {inputs};
}}",
                    kernel
                        .split("\n")
                        .map(|s| format!("\t{s}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            } else {
                kernel = format!(
                    "{kernel}
float {var_name} = {instruction}({inputs});",
                )
            }
            kernel = kernel.trim().to_string();
        }

        let output_index = get_inputs(&stack.frames).join(" + ");
        let last_var_name = (b'a' + ((var_names - 1) % 26) as u8) as char;
        let inputs = input_buffer_indexes
            .iter()
            .enumerate()
            .map(|(index, i)| {
                format!(
                    "\tdevice float* {} [[buffer({index})]]",
                    (b'A' + (i % 26) as u8) as char
                )
            })
            .join(",\n");
        kernel = kernel.split("\n").map(|k| format!("\t{k}")).join("\n");
        kernel = format!(
            "{PRELUDE}
kernel void kernel{n_kernel}(
{inputs},
	device float* out [[buffer({})]],
	uint3 blockIdx [[threadgroup_position_in_grid]],
	uint3 threadIdx [[thread_position_in_threadgroup]]
) {{
{kernel}
	out[{output_index}] = {last_var_name};
}}",
            input_buffer_indexes.len()
        );

        kernels.push(Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            inputs: input_buffer_indexes,
            outputs: vec![output_buffer_size],
        });
        logical_index_start_kernel += instructions.len();
    }

    kernels
}

fn get_inputs(frames: &[StackFrame]) -> Vec<String> {
    // Transpose from indexes[inputs[]] to inputs[indexes[]]
    let mut indexes = vec![vec![]; frames[0].input_strides.len()];
    for StackFrame {
        input_strides,
        reduce,
        ..
    } in frames
    {
        if !*reduce {
            for (i, s) in input_strides.iter().enumerate() {
                indexes[i].push(*s);
            }
        }
    }

    let index_to_dim = [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ];
    println!("Strides: {:?}", indexes);
    indexes
        .iter()
        .map(|stride| {
            let mut dim_index = 0;
            stride
                .iter()
                .enumerate()
                .flat_map(|(i, s)| {
                    if *s == 0 {
                        if frames[i].loop_char.is_none() {
                            dim_index += 1;
                        }
                        return None;
                    }
                    if let Some(l) = frames[i].loop_char {
                        Some(format!(
                            "loop_{l}{}",
                            if *s == 1 {
                                "".to_string()
                            } else {
                                format!(" * {s}")
                            }
                        ))
                    } else {
                        dim_index += 1;
                        Some(format!(
                            "{}{}",
                            index_to_dim[dim_index - 1],
                            if *s == 1 {
                                "".to_string()
                            } else {
                                format!(" * {s}")
                            }
                        ))
                    }
                })
                .collect::<Vec<_>>()
                .join(" + ")
        })
        .collect()
}
