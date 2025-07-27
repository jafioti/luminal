use cudarc::driver::{CudaSlice, CudaStream};
use itertools::Itertools;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::sync::Arc;

use luminal::prelude::*;

use crate::gguf::*;

pub fn load(
    path: impl AsRef<Path>,
    model: &impl SerializeModule,
) -> Vec<(NodeIndex, CudaSlice<f32>)> {
    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Load weights
    let mut weights = vec![];
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    for (weight_name, node_index) in param_dict(model) {
        // Read bytes
        let (n_elements, buffer_offset, data_type) =
            tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
        let n_bytes = match data_type {
            GgmlDType::F32 => n_elements * 4,
            GgmlDType::Q8_0 => (n_elements / 32) * 34,
            _ => panic!("unrecognized dtype {data_type:?}"),
        };
        let mut bytes = vec![0; n_bytes];
        let mut file = File::open(&path).unwrap();
        file.seek(std::io::SeekFrom::Start(
            buffer_offset as u64 + tensor_data_offset,
        ))
        .unwrap();
        file.read_exact(&mut bytes).unwrap();
        // Parse into data
        let data = match data_type {
            GgmlDType::F32 => bytes
                .into_iter()
                .chunks(4)
                .into_iter()
                .map(|c| {
                    let c = c.collect::<Vec<_>>();
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                })
                .collect::<Vec<_>>(),
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
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        weights.push((node_index, copy_cuda_buffer(&data, &stream)));
    }
    weights
}

pub fn copy_cuda_buffer(v: &Vec<f32>, stream: &Arc<CudaStream>) -> CudaSlice<f32> {
    let mut buffer = unsafe { stream.alloc::<f32>(v.len()).unwrap() };
    stream.memcpy_htod(v, &mut buffer).unwrap();
    buffer
}
