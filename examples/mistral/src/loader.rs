use std::{
    fs::File,
    io::{Read, Seek},
};

use luminal::{op::Function, prelude::*};

#[cfg(feature = "cuda")]
use {luminal_cuda::CudaData, luminal_cudarc::driver::CudaDevice};

use crate::gguf::*;

#[cfg(not(feature = "metal"))]
use itertools::Itertools;
#[cfg(feature = "metal")]
use {
    luminal_metal::MetalBuffer,
    memmap2::Mmap,
    metal_rs::{Device, MTLResourceOptions},
};

pub struct Q8Loader(String);

impl Q8Loader {
    pub fn new<S: Into<String>>(path: S) -> Self {
        Self(path.into())
    }
}

#[cfg(feature = "metal")]
impl Loader for Q8Loader {
    type Output = Vec<NodeIndex>;
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Output {
        // Read metadata from file
        let mut reader = File::open(&self.0).unwrap();
        let Content {
            mut tensor_infos,
            tensor_data_offset,
            ..
        } = Content::read(&mut reader).unwrap();

        // Create weight loading closures
        let mut q8_weights = vec![];
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_path = self.0.clone();
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
                    vec![Tensor {
                        data: Box::new(MetalBuffer(buffer)),
                    }]
                });
            }
        }
        q8_weights
    }
}

#[cfg(feature = "cuda")]
impl Loader for Q8Loader {
    type Output = Vec<NodeIndex>;
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Output {
        // Read metadata from file
        let mut reader = File::open(&self.0).unwrap();
        let Content {
            mut tensor_infos,
            tensor_data_offset,
            ..
        } = Content::read(&mut reader).unwrap();

        // Create weight loading closures
        let mut q8_weights = vec![];
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_path = self.0.clone();
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
                    let device = CudaDevice::new(0).unwrap();
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
                        GgmlDType::Q8_0 => vec![Tensor::new(CudaData(
                            device.htod_sync_copy::<u8>(&bytes).unwrap(),
                        ))],
                        _ => unimplemented!(),
                    }
                });
            }
        }
        q8_weights
    }
}

#[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
impl Loader for Q8Loader {
    type Output = Vec<NodeIndex>;
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> Self::Output {
        #[repr(C, packed)]
        #[derive(Clone, Copy)]
        struct Q8Block {
            delta: f16,
            weights: [i8; 32],
        }

        // Read metadata from file
        let mut reader = File::open(&self.0).unwrap();
        let Content {
            mut tensor_infos,
            tensor_data_offset,
            ..
        } = Content::read(&mut reader).unwrap();

        // Create weight loading closures
        let mut q8_weights = vec![];
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_path = self.0.clone();
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
}
