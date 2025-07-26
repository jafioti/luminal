use cudarc::driver::CudaSlice;
use itertools::Itertools;
use luminal_cuda::CudaData;
use memmap2::Mmap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use luminal::{op::Function, prelude::*};

use crate::gguf::*;

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

                        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
                        let stream = ctx.default_stream();
                        let mut buffer = unsafe { stream.alloc::<f32>(data.len()).unwrap() };
                        stream.memcpy_htod(&data, &mut buffer);
                        vec![Tensor::new(CudaData(buffer))]
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

                        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
                        let stream = ctx.default_stream();
                        let mut buffer = unsafe { stream.alloc::<f32>(data.len()).unwrap() };
                        stream.memcpy_htod(&data, &mut buffer);
                        vec![Tensor::new(CudaData(buffer))]
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
) -> Vec<(NodeIndex, Box<dyn FnOnce() -> CudaSlice<f32>>)> {
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
        match data_type {
            GgmlDType::F32 => {
                weights.push((
                    node_index,
                    Box::new(move || {
                        // Read bytes
                        let mut bytes = vec![0; n_elements * 4];
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
                        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
                        let stream = ctx.default_stream();
                        let mut buffer = unsafe { stream.alloc::<f32>(data.len()).unwrap() };
                        stream.memcpy_htod(&data, &mut buffer);
                        buffer
                    }) as Box<dyn FnOnce() -> CudaSlice<f32>>,
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
                        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
                        let stream = ctx.default_stream();
                        let mut buffer = unsafe { stream.alloc::<f32>(data.len()).unwrap() };
                        stream.memcpy_htod(&data, &mut buffer);
                        buffer
                    }),
                ));
            }
            _ => panic!("unrecognized dtype {:?}", data_type),
        }
    }
    weights
}
