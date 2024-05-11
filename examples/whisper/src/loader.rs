use std::fs::File;
use std::path::Path;

use crate::gguf::*;
use luminal::{op::Function, prelude::*};
use {
    luminal_metal::MetalBuffer,
    memmap2::Mmap,
    metal_rs::{Device, MTLResourceOptions},
};

pub fn load<P: AsRef<Path>, M: SerializeModule>(path: P, model: &M, graph: &mut Graph) {
    // Read metadata from file
    let mut reader = File::open(&path).unwrap();
    let Content {
        mut tensor_infos,
        tensor_data_offset,
        ..
    } = Content::read(&mut reader).unwrap();

    // Create weight loading closures
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
                _ => panic!("Unsupported dtype: {data_type:?}"),
            };
            loading_node.1 = Box::new(move |_| {
                let mmap_buffer = unsafe { Mmap::map(&File::open(&file_path).unwrap()).unwrap() };
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
