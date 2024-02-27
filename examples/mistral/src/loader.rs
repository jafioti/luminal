use std::fs::File;

use luminal::{op::Function, prelude::*};
use luminal_metal::MetalBuffer;
use memmap2::Mmap;
use metal_rs::{Device, MTLResourceOptions};

use crate::gguf::*;
pub struct MetalQ8Loader(String);
impl MetalQ8Loader {
    pub fn new<S: Into<String>>(path: S) -> Self {
        Self(path.into())
    }
}

impl Loader for MetalQ8Loader {
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
                    let buffer = Device::system_default().unwrap().new_buffer_with_data(
                        unsafe {
                            mmap_buffer
                                .as_ptr()
                                .add(buffer_offset + tensor_data_offset as usize)
                                as *const _
                        },
                        n_bytes as u64,
                        MTLResourceOptions::StorageModeShared,
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
