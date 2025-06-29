use std::collections::HashMap;

use itertools::Itertools;
use metal_rs::objc::rc::autoreleasepool;
use petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef};

use crate::Kernel;

pub fn run_graph(
    inputs: &[Vec<f32>],
    kernels: &StableGraph<Kernel, (u8, u8)>,
) -> (Vec<Vec<f32>>, u128) {
    use metal_rs::{
        CompileOptions, ComputePassDescriptor, ComputePipelineDescriptor, Device,
        MTLResourceOptions, MTLSize,
    };
    autoreleasepool(|| {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        // Allocate input buffers
        let mut buffers = HashMap::new();
        for node in toposort(kernels, None).unwrap() {
            let kernel = kernels.node_weight(node).unwrap();
            if kernel.code == "Inputs" {
                buffers.insert(
                    node,
                    inputs
                        .iter()
                        .map(|buf| {
                            device.new_buffer_with_data(
                                buf.as_ptr() as *mut _,
                                (buf.len() * std::mem::size_of::<f32>()) as u64,
                                MTLResourceOptions::StorageModeShared,
                            )
                        })
                        .collect_vec(),
                );
            } else if kernel.code == "Outputs" {
                // Run
                let start = std::time::Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let time_taken_micros = start.elapsed().as_micros();

                let outputs = kernels
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| &buffers[&e.source()][e.weight().0 as usize])
                    .map(|buffer| {
                        let mut curr_data =
                            vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
                        let ptr = buffer.contents() as *mut f32;
                        for (i, d) in curr_data.iter_mut().enumerate() {
                            *d = unsafe { *ptr.add(i) };
                        }
                        curr_data
                    })
                    .collect();

                // Copy outputs back
                return (outputs, time_taken_micros);
            } else {
                // allocate output buffers
                let outputs = kernel
                    .outputs
                    .iter()
                    .map(|size| {
                        device.new_buffer(
                            (size.to_usize().unwrap() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    })
                    .collect_vec();
                buffers.insert(node, outputs);

                // compile kernel
                let encoder = command_buffer
                    .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
                let options = CompileOptions::new();
                options.set_fast_math_enabled(true);
                let lib = device
                    .new_library_with_source(&kernel.code, &options)
                    .unwrap();
                let pipeline_state_descriptor = ComputePipelineDescriptor::new();
                pipeline_state_descriptor.set_compute_function(Some(
                    &lib.get_function(&format!("kernel{}", node.index()), None)
                        .unwrap(),
                ));
                let pipeline = device
                    .new_compute_pipeline_state_with_function(
                        pipeline_state_descriptor.compute_function().unwrap(),
                    )
                    .unwrap();
                encoder.set_compute_pipeline_state(&pipeline);

                // set inputs
                for (i, (input, input_index)) in kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                    .enumerate()
                {
                    encoder.set_buffer(i as u64, Some(&buffers[&input][input_index as usize]), 0);
                }
                // set output
                let n_inputs = kernels.edges_directed(node, Direction::Incoming).count();
                for (i, output) in buffers[&node].iter().enumerate() {
                    encoder.set_buffer((i + n_inputs) as u64, Some(output), 0);
                }
                // set smem
                if !kernel.smem.is_empty() {
                    encoder.set_threadgroup_memory_length(
                        0,
                        (kernel.smem.to_usize().unwrap() * std::mem::size_of::<f32>()) as u64,
                    );
                }

                // Set dispatch
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        kernel.grid.0.to_usize().unwrap() as u64,
                        kernel.grid.1.to_usize().unwrap() as u64,
                        kernel.grid.2.to_usize().unwrap() as u64,
                    ),
                    MTLSize::new(
                        kernel.threadblock.0.to_usize().unwrap() as u64,
                        kernel.threadblock.1.to_usize().unwrap() as u64,
                        kernel.threadblock.2.to_usize().unwrap() as u64,
                    ),
                );
                encoder.end_encoding();
            }
        }
        panic!("No output kernel detected in graph!");
    })
}
