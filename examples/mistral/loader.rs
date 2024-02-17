use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use itertools::Itertools;
use luminal::{op::Function, prelude::*};
use memmap2::Mmap;
use metal_rs::{Device, MTLResourceOptions};
use safetensors::SafeTensors;

use crate::gguf::*;

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
    type Output = ();
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) -> () {
        // Read metadata from file
        let mut reader =
            File::open("/Users/jafioti/Downloads/mistral-7b-instruct-v0.2.Q8_0.gguf").unwrap();
        let Content {
            mut tensor_infos,
            tensor_data_offset,
            ..
        } = Content::read(&mut reader).unwrap();
        for (weight_name, node_index) in state_dict(model) {
            if let Some(loading_node) = graph
                .graph
                .node_weight_mut(node_index)
                .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
            {
                let (n_elements, buffer_offset, data_type) =
                    tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
                loading_node.1 = match data_type {
                    GgmlDType::F32 => {
                        Box::new(move |_| {
                            let mut file = File::open(
                                "/Users/jafioti/Downloads/mistral-7b-instruct-v0.2.Q8_0.gguf",
                            )
                            .unwrap();
                            file.seek(SeekFrom::Start(tensor_data_offset + buffer_offset as u64))
                                .unwrap();
                            // Load buffer and do conversion
                            let mut byte_buffer = vec![0; n_elements * 4];
                            file.read_exact(&mut byte_buffer).unwrap();
                            let f32_buffer = byte_buffer
                                .into_iter()
                                // Group into f32s
                                .chunks(4)
                                .into_iter()
                                // Convert to f16s
                                .map(|chunk| {
                                    let bytes = chunk.collect::<Vec<_>>();
                                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                                })
                                .collect::<Vec<_>>();
                            vec![Tensor {
                                data: Box::new(f32_buffer),
                            }]
                            // let mmap_buffer = unsafe { Mmap::map(&file).unwrap() };
                            // let buffer = Device::system_default()
                            //     .unwrap()
                            //     .new_buffer_with_bytes_no_copy(
                            //         mmap_buffer.as_ptr() as *const _,
                            //         (n_elements + (n_elements / 16)) as u64,
                            //         MTLResourceOptions::StorageModeShared,
                            //         None,
                            //     );
                            // vec![Tensor {
                            //     data: Box::new(buffer),
                            // }]
                        })
                    }
                    GgmlDType::Q8_0 => {
                        Box::new(move |_| {
                            #[repr(C, packed)]
                            struct BlockQ8_0 {
                                _d: f16,
                                _qs: [i8; 32],
                            }
                            let mut bytes = vec![0; n_elements + (n_elements / 16)];
                            let mut file = File::open(
                                "/Users/jafioti/Downloads/mistral-7b-instruct-v0.2.Q8_0.gguf",
                            )
                            .unwrap();
                            file.seek(SeekFrom::Start(tensor_data_offset + buffer_offset as u64))
                                .unwrap();
                            file.read_exact(&mut bytes).unwrap();
                            let blocks = bytes
                                .into_iter()
                                .chunks(34)
                                .into_iter()
                                .map(|chunk| {
                                    let chunk = chunk.collect::<Vec<_>>();
                                    let mut bytes = [0; 34];
                                    bytes.copy_from_slice(&chunk);
                                    unsafe { std::mem::transmute::<_, BlockQ8_0>(bytes) }
                                })
                                .collect::<Vec<_>>();
                            let f32_buffer = blocks
                                .into_iter()
                                .flat_map(|b| {
                                    b._qs.into_iter().map(move |i| {
                                        let r = i as f32 * b._d.to_f32();
                                        let l = b._d;
                                        assert!(!r.is_nan(), "{i} and {}", l);
                                        r
                                    })
                                })
                                .collect::<Vec<_>>();
                            vec![Tensor {
                                data: Box::new(f32_buffer),
                            }]

                            // f32_buffer.leak();
                            // let buffer = Device::system_default()
                            //     .unwrap()
                            //     .new_buffer_with_bytes_no_copy(
                            //         mmap_buffer.as_ptr() as *const _,
                            //         (n_elements + (n_elements / 16)) as u64,
                            //         MTLResourceOptions::StorageModeShared,
                            //         None,
                            //     );
                            // vec![Tensor {
                            //     data: Box::new(buffer),
                            // }]
                            // vec![Tensor {
                            //     data: Box::new(buffer),
                            // }]
                        })
                    }
                    _ => panic!("Unsupported dtype"),
                };

                // let file_paths = self.paths.clone();
                // let weight_name = weight_name
                //     .replace("ffn_gate", "mlp/gate_proj")
                //     .replace("ffn_up", "mlp/up_proj")
                //     .replace("ffn_down", "mlp/down_proj")
                //     .replace("attn_q", "self_attn/q_proj")
                //     .replace("attn_k", "self_attn/k_proj")
                //     .replace("attn_v", "self_attn/v_proj")
                //     .replace("attn_output", "self_attn/o_proj")
                //     .replace("attn_norm", "input_layernorm")
                //     .replace("ffn_norm", "post_attention_layernorm")
                //     .replace("blk", "model/layers")
                //     .replace("token_embd", "model/embed_tokens")
                //     .replace("output_norm", "model/norm")
                //     .replace("output", "lm_head");
                // loading_node.1 = Box::new(move |_| {
                //     for file_path in file_paths.iter() {
                //         let file = File::open(file_path).unwrap();
                //         let buffer = unsafe { Mmap::map(&file).unwrap() };
                //         let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                //         if let Ok(tensor_view) = safetensors.tensor(&weight_name.replace('/', "."))
                //         {
                //             let buffer = Device::system_default()
                //                 .unwrap()
                //                 .new_buffer_with_bytes_no_copy(
                //                     tensor_view.data().as_ptr() as *const _,
                //                     tensor_view.data().len() as u64,
                //                     MTLResourceOptions::StorageModeShared,
                //                     None,
                //                 );
                //             return vec![Tensor {
                //                 data: Box::new(buffer),
                //             }];
                //         }
                //     }

                //     panic!("Tensor \"{weight_name}\" not found in files");
                // });
            }
        }
    }
}

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
                // Special embed case
                if weight_name.contains("fewkferk") {
                    let weight_name = weight_name
                        .replace("ffn_gate", "mlp/gate_proj")
                        .replace("ffn_up", "mlp/up_proj")
                        .replace("ffn_down", "mlp/down_proj")
                        .replace("attn_q", "self_attn/q_proj")
                        .replace("attn_k", "self_attn/k_proj")
                        .replace("attn_v", "self_attn/v_proj")
                        .replace("attn_output", "self_attn/o_proj")
                        .replace("attn_norm", "input_layernorm")
                        .replace("ffn_norm", "post_attention_layernorm")
                        .replace("blk", "model/layers")
                        .replace("token_embd", "model/embed_tokens")
                        .replace("output_norm", "model/norm")
                        .replace("output", "lm_head");
                    loading_node.1 = Box::new(move |_| {
                        println!("Weight: {weight_name}");
                        for file_path in ["./examples/mistral/setup/mistral-7b-hf/converted-model-00001-of-00003.safetensors",
                            "./examples/mistral/setup/mistral-7b-hf/converted-model-00002-of-00003.safetensors",
                            "./examples/mistral/setup/mistral-7b-hf/converted-model-00003-of-00003.safetensors",] {
                            let file = File::open(file_path).unwrap();
                            let buffer = unsafe { Mmap::map(&file).unwrap() };
                            let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                            if let Ok(tensor_view) =
                                safetensors.tensor(&weight_name.replace('/', "."))
                            {
                                let fp32_buffer = tensor_view.data().bytes().chunks(2).into_iter().map(|bytes| {
                                    let chunk = bytes.flatten().collect::<Vec<_>>();
                                    f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                                }).collect::<Vec<_>>();
                                #[repr(C, packed)]
                                #[derive(Debug)]
                                struct BlockQ8_0 {
                                    _d: f16,
                                    _qs: [i8; 32],
                                }
                                let q8_blocks = fp32_buffer.into_iter().chunks(32).into_iter().map(|chunk| {
                                    let chunk = chunk.collect::<Vec<_>>();
                                    let amax = chunk.iter().map(|i| i.abs()).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
                                    let d = amax / ((1 << 7) - 1) as f32;
                                    let id = if d == 0.0 {0.0} else {1. / d};
                                    let mut q = [0; 32];
                                    for i in 0..32 {
                                        q[i] = (chunk[i] * id).round() as i8;
                                    }
                                    if q.iter().any(|i| (*i as f32 * d).is_nan()) {
                                        panic!("We got one");
                                    }
                                    BlockQ8_0 {
                                        _d: f16::from_f32(d),
                                        _qs: q
                                    }
                                }).collect::<Vec<_>>();
                                println!("Block: {:?}", q8_blocks[0]);
                                let buffer = Device::system_default()
                                    .unwrap()
                                    .new_buffer_with_data(
                                        q8_blocks.as_ptr() as *const _,
                                        (q8_blocks.len() * 34) as u64,
                                        MTLResourceOptions::StorageModeShared,
                                    );
                                return vec![Tensor {
                                    data: Box::new(buffer),
                                }];
                            }
                        }

                        panic!("Tensor \"{weight_name}\" not found in files");
                    });
                    continue;
                }
                if weight_name.contains("token_embd") {
                    let weight_name = weight_name
                        .replace("ffn_gate", "mlp/gate_proj")
                        .replace("ffn_up", "mlp/up_proj")
                        .replace("ffn_down", "mlp/down_proj")
                        .replace("attn_q", "self_attn/q_proj")
                        .replace("attn_k", "self_attn/k_proj")
                        .replace("attn_v", "self_attn/v_proj")
                        .replace("attn_output", "self_attn/o_proj")
                        .replace("attn_norm", "input_layernorm")
                        .replace("ffn_norm", "post_attention_layernorm")
                        .replace("blk", "model/layers")
                        .replace("token_embd", "model/embed_tokens")
                        .replace("output_norm", "model/norm")
                        .replace("output", "lm_head");
                    loading_node.1 = Box::new(move |_| {
                        for file_path in ["./examples/mistral/setup/mistral-7b-hf/converted-model-00001-of-00003.safetensors",
                            "./examples/mistral/setup/mistral-7b-hf/converted-model-00002-of-00003.safetensors",
                            "./examples/mistral/setup/mistral-7b-hf/converted-model-00003-of-00003.safetensors",] {
                            let file = File::open(file_path).unwrap();
                            let buffer = unsafe { Mmap::map(&file).unwrap() };
                            let safetensors = SafeTensors::deserialize(&buffer).unwrap();

                            if let Ok(tensor_view) =
                                safetensors.tensor(&weight_name.replace('/', "."))
                            {
                                let fp32_buffer = tensor_view.data().bytes().take(tensor_view.data().len()).chunks(2).into_iter().map(|bytes| {
                                    let chunk = bytes.flatten().collect::<Vec<_>>();
                                    f16::from_le_bytes([chunk[0], chunk[1]])
                                }).collect::<Vec<_>>();
                                let buffer = Device::system_default()
                                    .unwrap()
                                    .new_buffer_with_bytes_no_copy(
                                        fp32_buffer.as_ptr() as *const _,
                                        (fp32_buffer.len() * 2) as u64,
                                        MTLResourceOptions::StorageModeShared,
                                        None,
                                    );
                                fp32_buffer.leak();
                                return vec![Tensor {
                                    data: Box::new(buffer),
                                }];
                            }
                        }

                        panic!("Tensor \"{weight_name}\" not found in files");
                    });
                    continue;
                }
                let file_path = self.0.clone();
                let (n_elements, buffer_offset, data_type) =
                    tensor_infos.remove(&weight_name.replace('/', ".")).unwrap();
                loading_node.1 = match data_type {
                    GgmlDType::F32 => Box::new(move |_| {
                        let mut file = File::open(&file_path).unwrap();
                        file.seek(SeekFrom::Start(tensor_data_offset + buffer_offset as u64))
                            .unwrap();
                        // Load buffer and do conversion
                        let mut byte_buffer = vec![0; n_elements * 4];
                        file.read_exact(&mut byte_buffer).unwrap();
                        let f32_buffer = byte_buffer
                            .into_iter()
                            // Group into f32s
                            .chunks(4)
                            .into_iter()
                            // Convert to f16s
                            .map(|chunk| {
                                let bytes = chunk.collect::<Vec<_>>();
                                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            })
                            .collect::<Vec<_>>();
                        vec![Tensor {
                            data: Box::new(f32_buffer),
                        }]
                    }),
                    GgmlDType::Q8_0 => {
                        q8_weights.push(node_index);
                        Box::new(move |_| {
                            let mmap_buffer =
                                unsafe { Mmap::map(&File::open(&file_path).unwrap()).unwrap() };
                            let buffer = Device::system_default().unwrap().new_buffer_with_data(
                                unsafe {
                                    mmap_buffer
                                        .as_ptr()
                                        .add(buffer_offset + tensor_data_offset as usize)
                                        as *const _
                                },
                                (n_elements + (n_elements / 16)) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            vec![Tensor {
                                data: Box::new(buffer),
                            }]
                        })
                    }
                    _ => panic!("Unsupported dtype"),
                };
            }
        }
        q8_weights
    }
}
