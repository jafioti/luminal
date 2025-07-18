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
                        vec![Tensor::new(
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
                                .collect::<Vec<_>>(),
                        )]
                    });
                }
                _ => panic!("unrecognized dtype {:?}", data_type),
            }
        }
    }
    q8_weights
}

pub fn q8_load_new<P: AsRef<Path>, M: SerializeModule>(
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

pub fn load_param_f32<P: AsRef<Path>>(path: P, param_name: &str) -> Vec<f32> {
    // Open the file and read content metadata
    let mut file = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut file).unwrap();

    // Map parameter name to file format (dot vs slash etc)
    let key = param_name.replace('/', ".");

    // Get (n_elements, buffer_offset, dtype)
    let (n_elements, buffer_offset, dtype) = tensor_infos
        .remove(&key)
        .expect("Parameter not found in file");

    // Read appropriate number of bytes
    let n_bytes = match dtype {
        GgmlDType::F32 => n_elements * 4,
        GgmlDType::Q8_0 => n_elements / 32 * 34, // Each 16 i8 + f32 = 20 bytes for 16 values
        _ => panic!("Unsupported dtype: {:?}", dtype),
    };

    // Seek to the parameter's data
    file.seek(SeekFrom::Start(buffer_offset as u64 + tensor_data_offset))
        .unwrap();

    let mut bytes = vec![0u8; n_bytes];
    file.read_exact(&mut bytes).unwrap();

    match dtype {
        GgmlDType::F32 => {
            // Just convert chunks of 4 bytes to f32
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        GgmlDType::Q8_0 => bytes
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
            .collect(),
        _ => panic!("Unsupported dtype: {:?}", dtype),
    }
}
