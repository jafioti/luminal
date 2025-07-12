#[allow(unused_imports)]
use std::io::Read;
#[allow(unused_imports)]
use std::path::Path;
#[allow(unused_imports)]
use std::{fs::File, io::Seek};

use luminal::{op::Function, prelude::*};
#[allow(unused_imports)]
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};

pub fn load<M: SerializeModule>(path: &str, model: &M, graph: &mut Graph) {
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            let path = path.to_string();
            loading_node.1 = Box::new(move |_| {
                let file = File::open(&path).unwrap();
                let mmap = unsafe { Mmap::map(&file).unwrap() };
                let safetensors = SafeTensors::deserialize(&mmap).unwrap();

                if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', ".")) {
                    let data: Vec<f32> = match tensor_view.dtype() {
                        Dtype::F32 => tensor_view
                            .data()
                            .chunks_exact(4)
                            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        Dtype::F16 => tensor_view
                            .data()
                            .chunks_exact(2)
                            .map(|c| f16::from_ne_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        _ => panic!("{:?} is not a supported dtype", tensor_view.dtype()),
                    };
                    return vec![Tensor::new(data)];
                }

                panic!("Tensor \"{weight_name}\" not found in files");
            });
        }
    }
}
