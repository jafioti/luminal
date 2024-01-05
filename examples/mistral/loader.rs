use std::fs::File;

use luminal::{op::Function, prelude::*};
use memmap2::MmapOptions;
use metal_rs::{Device, MTLResourceOptions};
use safetensors::SafeTensors;

/// Load the model in the same way dfdx-llama does
pub struct MetalFp16SafetensorsLoader {
    paths: Vec<String>,
}

impl MetalFp16SafetensorsLoader {
    pub fn new<S: ToString>(paths: &[S]) -> Self {
        Self {
            paths: paths.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Loader for MetalFp16SafetensorsLoader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let file_paths = self.paths.clone();
                loading_node.1 = Box::new(move |_| {
                    for file_path in file_paths.iter() {
                        let file = File::open(file_path).unwrap();
                        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
                        let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                        if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', "."))
                        {
                            let buffer = Device::system_default()
                                .unwrap()
                                .new_buffer_with_bytes_no_copy(
                                    tensor_view.data().as_ptr() as *const _,
                                    tensor_view.data().len() as u64,
                                    MTLResourceOptions::StorageModeShared,
                                    None,
                                );
                            return vec![Tensor {
                                data: Box::new(buffer),
                            }];
                        }
                    }

                    panic!("Tensor \"{weight_name}\" not found in files");
                });
            }
        }
    }
}
