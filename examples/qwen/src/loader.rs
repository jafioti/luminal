use itertools::Itertools;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use luminal::{op::Function, prelude::*};

#[cfg(feature = "cuda")]
use luminal_cuda::{CudaContext, CudaData};

use crate::gguf::*;

#[cfg(feature = "metal")]
use {
    luminal_metal::{Device, MTLResourceOptions, MetalBuffer},
    memmap2::Mmap,
};

#[cfg(feature = "metal")]
pub fn q8_load<P: AsRef<Path>, M: SerializeModule>(
    path: P,
    model: &M,
    graph: &mut Graph,
) -> Vec<NodeIndex> {
    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Create weight loading closures
    let mut q8_weights = vec![];
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            let file_path = path.as_ref().to_owned();
            let (n_elements, buffer_offset, data_type) = tensor_infos
                .remove(&weight_name.replace('/', "."))
                .unwrap_or_else(|| panic!("Couldn't find weight {weight_name}"));
            let n_bytes = match data_type {
                GgmlDType::F32 => n_elements * 4,
                GgmlDType::Q8_0 => {
                    q8_weights.push(node_index);
                    n_elements + (n_elements / 16)
                }
                _ => panic!("Unsupported dtype: {data_type:?}"),
            };
            if let GgmlDType::F32 = data_type {
                loading_node.1 = Box::new(move |_| {
                    // Read bytes
                    let mut bytes = vec![0; n_bytes];
                    let mut file = File::open(&file_path).unwrap();
                    file.seek(std::io::SeekFrom::Start(
                        buffer_offset as u64 + tensor_data_offset,
                    ))
                    .unwrap();
                    file.read_exact(&mut bytes).unwrap();
                    vec![Tensor::new(
                        bytes
                            .into_iter()
                            .chunks(4)
                            .into_iter()
                            .map(|c| {
                                let c = c.collect::<Vec<_>>();
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            })
                            .collect::<Vec<_>>(),
                    )]
                });
            } else {
                loading_node.1 = Box::new(move |_| {
                    let mmap_buffer =
                        unsafe { Mmap::map(&File::open(&file_path).unwrap()).unwrap() };
                    let buffer = Device::system_default()
                        .unwrap()
                        .new_buffer_with_bytes_no_copy(
                            unsafe {
                                mmap_buffer
                                    .as_ptr()
                                    .add(buffer_offset + tensor_data_offset as usize)
                                    as *const _
                            },
                            n_bytes as u64,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        );
                    vec![Tensor::new(MetalBuffer(buffer))]
                });
            }
        }
    }
    q8_weights
}

#[cfg(feature = "cuda")]
pub fn q8_load<P: AsRef<Path>, M: SerializeModule>(
    path: P,
    model: &M,
    graph: &mut Graph,
) -> Vec<NodeIndex> {
    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Create weight loading closures
    let mut q8_weights = vec![];
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            let file_path = path.as_ref().to_owned();
            let (n_elements, buffer_offset, data_type) =
                tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
            let n_bytes = match data_type {
                GgmlDType::F32 => n_elements * 4,
                GgmlDType::Q8_0 => {
                    q8_weights.push(node_index);
                    n_elements + (n_elements / 16)
                }
                _ => panic!("Unsupported dtype: {data_type:?}"),
            };
            loading_node.1 = Box::new(move |_| {
                // Read bytes
                let mut bytes = vec![0; n_bytes];
                let mut file = File::open(&file_path).unwrap();
                file.seek(std::io::SeekFrom::Start(
                    buffer_offset as u64 + tensor_data_offset,
                ))
                .unwrap();
                file.read_exact(&mut bytes).unwrap();
                // Copy buffer over to cuda slice
                let device = CudaContext::new(0).unwrap();
                let stream = device.default_stream();
                match data_type {
                    GgmlDType::F32 => vec![Tensor::new(
                        bytes
                            .into_iter()
                            .chunks(4)
                            .into_iter()
                            .map(|c| {
                                let c = c.collect::<Vec<_>>();
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            })
                            .collect::<Vec<_>>(),
                    )],
                    GgmlDType::Q8_0 => {
                        vec![Tensor::new(CudaData(stream.memcpy_stod(&bytes).unwrap()))]
                    }
                    _ => unimplemented!(),
                }
            });
        }
    }
    q8_weights
}

#[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
pub fn q8_load<P: AsRef<Path>, M: SerializeModule>(
    path: P,
    model: &M,
    graph: &mut Graph,
) -> Vec<NodeIndex> {
    #[repr(C, packed)]
    #[derive(Clone, Copy)]
    struct Q8Block {
        delta: f16,
        weights: [i8; 32],
    }

    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Create weight loading closures
    let mut q8_weights = vec![];
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            let file_path = path.as_ref().to_owned();
            let (n_elements, buffer_offset, data_type) =
                tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
            let n_bytes = match data_type {
                GgmlDType::F32 => n_elements * 4,
                GgmlDType::Q8_0 => {
                    q8_weights.push(node_index);
                    n_elements + (n_elements / 16)
                }
                _ => panic!("Unsupported dtype: {data_type:?}"),
            };
            loading_node.1 = Box::new(move |_| {
                // Load all bytes
                let mut bytes = vec![0; n_bytes];
                let mut file = File::open(&file_path).unwrap();
                file.seek(std::io::SeekFrom::Start(
                    buffer_offset as u64 + tensor_data_offset,
                ))
                .unwrap();
                file.read_exact(&mut bytes).unwrap();
                // Dequantize into f32
                let data: Vec<f32> = match data_type {
                    GgmlDType::F32 => bytes
                        .into_iter()
                        .chunks(4)
                        .into_iter()
                        .map(|c| {
                            let c = c.collect::<Vec<_>>();
                            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        })
                        .collect(),
                    GgmlDType::Q8_0 => bytes
                        .into_iter()
                        .chunks(34)
                        .into_iter()
                        .map(|c| {
                            let chunk = c.collect::<Vec<_>>();
                            unsafe { chunk.align_to::<Q8Block>().1[0] }
                        })
                        .flat_map(|chunk| {
                            chunk
                                .weights
                                .into_iter()
                                .map(move |i| i as f32 * chunk.delta.to_f32())
                        })
                        .collect(),
                    _ => panic!("Unsupported dtype: {data_type:?}"),
                };
                vec![Tensor::new(data)]
            });
        }
    }
    q8_weights
}
