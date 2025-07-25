use itertools::Itertools;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use luminal::{op::Function, prelude::*};

use crate::gguf::*;

use {
    luminal_metal::{Device, MTLResourceOptions, MetalBuffer},
    memmap2::Mmap,
};

pub fn load<P: AsRef<Path>, M: SerializeModule>(
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
                GgmlDType::Q8_0 => n_elements * 4,
                _ => panic!("Unsupported dtype: {data_type:?}"),
            };
            match data_type {
                GgmlDType::F32 => {
                    loading_node.1 = Box::new(move |_| {
                        // Read bytes
                        let mut bytes = vec![0; n_bytes];
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(std::io::SeekFrom::Start(
                            buffer_offset as u64 + tensor_data_offset,
                        ))
                        .unwrap();
                        file.read_exact(&mut bytes).unwrap();
                        let data = bytes
                            .into_iter()
                            .chunks(4)
                            .into_iter()
                            .map(|c| {
                                let c = c.collect::<Vec<_>>();
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            })
                            .collect::<Vec<_>>();
                        let buffer = Device::system_default().unwrap().new_buffer_with_data(
                            data.as_ptr() as *mut _,
                            (data.len() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        vec![Tensor::new(MetalBuffer(buffer))]
                    });
                }
                GgmlDType::Q8_0 => {
                    loading_node.1 = Box::new(move |_| {
                        // Read bytes
                        let mut bytes = vec![0; (n_elements / 32) * 34];
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(std::io::SeekFrom::Start(
                            buffer_offset as u64 + tensor_data_offset,
                        ))
                        .unwrap();
                        file.read_exact(&mut bytes).unwrap();
                        let data = bytes
                            .chunks_exact(32 + 2)
                            .into_iter()
                            .flat_map(|bytes| {
                                let delta = f16::from_ne_bytes([bytes[0], bytes[1]]).to_f32();
                                bytes
                                    .iter()
                                    .skip(2)
                                    .take(32)
                                    .map(move |b| i8::from_ne_bytes([*b]) as f32 * delta)
                            })
                            .collect::<Vec<_>>();
                        let buffer = Device::system_default().unwrap().new_buffer_with_data(
                            data.as_ptr() as *mut _,
                            (data.len() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        vec![Tensor::new(MetalBuffer(buffer))]
                    });
                }
                _ => panic!("unrecognized dtype {:?}", data_type),
            }
        }
    }
    q8_weights
}

pub fn load_new<P: AsRef<Path>, M: SerializeModule>(
    path: P,
    model: &M,
) -> Vec<(NodeIndex, Box<dyn FnOnce() -> Vec<f32>>)> {
    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Create weight loading closures
    let mut weights = vec![];
    for (weight_name, node_index) in param_dict(model) {
        let file_path = path.as_ref().to_owned();
        let (n_elements, buffer_offset, data_type) =
            tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
        let n_bytes = match data_type {
            GgmlDType::F32 => n_elements * 4,
            GgmlDType::Q8_0 => n_elements * 4,
            _ => panic!("Unsupported dtype: {data_type:?}"),
        };
        match data_type {
            GgmlDType::F32 => {
                weights.push((
                    node_index,
                    Box::new(move || {
                        // Read bytes
                        let mut bytes = vec![0; n_bytes];
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(std::io::SeekFrom::Start(
                            buffer_offset as u64 + tensor_data_offset,
                        ))
                        .unwrap();
                        file.read_exact(&mut bytes).unwrap();
                        bytes
                            .into_iter()
                            .chunks(4)
                            .into_iter()
                            .map(|c| {
                                let c = c.collect::<Vec<_>>();
                                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                            })
                            .collect::<Vec<_>>()
                    }) as Box<dyn FnOnce() -> Vec<f32>>,
                ));
            }
            GgmlDType::Q8_0 => {
                weights.push((
                    node_index,
                    Box::new(move || {
                        // Read bytes
                        let mut bytes = vec![0; (n_elements / 32) * 34];
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(std::io::SeekFrom::Start(
                            buffer_offset as u64 + tensor_data_offset,
                        ))
                        .unwrap();
                        file.read_exact(&mut bytes).unwrap();
                        bytes
                            .chunks_exact(32 + 2)
                            .into_iter()
                            .flat_map(|bytes| {
                                let delta = f16::from_ne_bytes([bytes[0], bytes[1]]).to_f32();
                                bytes
                                    .iter()
                                    .skip(2)
                                    .take(32)
                                    .map(move |b| i8::from_ne_bytes([*b]) as f32 * delta)
                            })
                            .collect::<Vec<_>>()
                    }),
                ));
            }
            _ => panic!("unrecognized dtype {:?}", data_type),
        }
    }
    weights
}

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
                GgmlDType::Q8_0 => n_elements + (n_elements / 16),
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
                if weight_name == "token_embd/weight" {
                    loading_node.1 = Box::new(move |_| {
                        // Read bytes
                        let mut bytes = vec![0; (n_elements / 32) * 34];
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(std::io::SeekFrom::Start(
                            buffer_offset as u64 + tensor_data_offset,
                        ))
                        .unwrap();
                        file.read_exact(&mut bytes).unwrap();
                        let data = bytes
                            .chunks_exact(32 + 2)
                            .into_iter()
                            .flat_map(|bytes| {
                                let delta = f16::from_ne_bytes([bytes[0], bytes[1]]).to_f32();
                                bytes
                                    .iter()
                                    .skip(2)
                                    .take(32)
                                    .map(move |b| i8::from_ne_bytes([*b]) as f32 * delta)
                            })
                            .collect::<Vec<_>>();
                        let buffer = Device::system_default().unwrap().new_buffer_with_data(
                            data.as_ptr() as *mut _,
                            (data.len() * std::mem::size_of::<f32>()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        vec![Tensor::new(MetalBuffer(buffer))]
                    });
                } else {
                    q8_weights.push(node_index);
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
    }
    q8_weights
}
