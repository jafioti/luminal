use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use itertools::Itertools;
use safetensors::SafeTensors;
use std::path::Path;
use std::{ffi::c_void, fs::File};

use luminal::{op::Function, prelude::*};

use luminal_metal::{Device, MTLResourceOptions, MetalBuffer};
use memmap2::Mmap;

/// Load weights from a Hugging Face model with SafeTensors format into a Luminal graph.
/// This handles distributed SafeTensors files (multiple .safetensors files with an index.json).
///
/// Arguments:
///   - model_name: Hugging Face model name (e.g., "facebook/opt-350m")
///   - model: Luminal model structure
///   - graph: Luminal graph to add weights to
///   - revision: Optional model revision/tag (defaults to "main")
///
/// Returns a vector of node indices for any quantized weights detected
pub fn load_from_huggingface<M: SerializeModule>(
    model_name: &str,
    model: &M,
    graph: &mut Graph,
    revision: Option<&str>,
) -> Vec<NodeIndex> {
    // Initialize Hugging Face API
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(
        model_name.to_string(),
        RepoType::Model,
        revision.unwrap_or("main").to_string(),
    ));

    // Download and read the model's index file
    let index_path = repo.get("model.safetensors.index.json").unwrap();
    let index_content = std::fs::read_to_string(index_path).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_content).unwrap();

    // Create a mapping from tensor name to file path
    let mut tensor_to_file = std::collections::HashMap::new();
    if let Some(weight_map) = index["weight_map"].as_object() {
        for (tensor_name, file_value) in weight_map {
            if let Some(file_name) = file_value.as_str() {
                tensor_to_file.insert(tensor_name.clone(), file_name.to_string());
            }
        }
    } else {
        // If no index found, check for a single safetensors file
        let safetensors_path = repo.get("model.safetensors").unwrap_or_else(|_| {
            panic!(
                "No model.safetensors file or index found for {}",
                model_name
            )
        });
        let tensor_names = {
            let file = File::open(&safetensors_path).unwrap();
            let buffer = unsafe { Mmap::map(&file).unwrap() };
            let tensors = SafeTensors::deserialize(&buffer).unwrap();
            tensors
                .names()
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>()
        };

        for tensor_name in tensor_names {
            tensor_to_file.insert(tensor_name, "model.safetensors".to_string());
        }
    }
    // println!("{:?}", tensor_to_file.keys().collect::<Vec<_>>());

    // Cache for file paths to avoid repeated downloads
    let mut file_cache = std::collections::HashMap::new();

    // Track quantized weights
    let mut quantized_weights = vec![];

    // Process each parameter in the model
    for (weight_name, node_index) in param_dict(model) {
        if let Some(loading_node) = graph
            .graph
            .node_weight_mut(node_index)
            .and_then(|op| op.as_any_mut().downcast_mut::<Function>())
        {
            // Convert parameter name format from model to safetensors format
            let safetensors_name = weight_name.replace('/', ".");

            // Find which file contains this tensor
            if let Some(file_name) = tensor_to_file.get(&safetensors_name) {
                // Download file if not already in cache
                if !file_cache.contains_key::<String>(file_name) {
                    let file_path = repo.get(file_name).unwrap();
                    file_cache.insert(file_name.clone(), file_path);
                }

                let file_path = file_cache.get(file_name).unwrap().clone();
                let sn = safetensors_name.clone();

                // Create a closure to load the tensor
                loading_node.1 = Box::new(move |_| {
                    let file = File::open(&file_path).unwrap();
                    let buffer = unsafe { Mmap::map(&file).unwrap() };
                    let tensors = SafeTensors::deserialize(&buffer).unwrap();

                    if let Ok(tensor) = tensors.tensor(&sn) {
                        let dtype = tensor.dtype();

                        match dtype {
                            // safetensors::Dtype::F32 => {
                            //     let data: Vec<f32> =
                            //         tensor.get_tensor_view::<f32>().unwrap().to_vec();
                            //     vec![Tensor::new(data)]
                            // }
                            safetensors::Dtype::F16 => {
                                let raw_data = tensor.data();
                                let data: Vec<f32> = raw_data
                                    .chunks_exact(2)
                                    .map(|bytes| {
                                        half::f16::from_ne_bytes([bytes[0], bytes[1]]).to_f32()
                                    })
                                    .collect();
                                vec![Tensor::new(data)]
                            }
                            safetensors::Dtype::BF16 => {
                                let raw_data = tensor.data();
                                let data: Vec<f32> = raw_data
                                    .chunks_exact(2)
                                    .map(|bytes| {
                                        // Convert BF16 to F32
                                        bf16::from_ne_bytes([bytes[0], bytes[1]]).to_f32()
                                    })
                                    .collect();
                                vec![Tensor::new(data)]
                            }
                            _ => {
                                // For quantized or other formats, use Metal buffer directly
                                let raw_data = tensor.data();
                                let buffer =
                                    Device::system_default().unwrap().new_buffer_with_data(
                                        raw_data.as_ptr() as *const c_void,
                                        raw_data.len() as u64,
                                        MTLResourceOptions::StorageModeShared,
                                    );
                                vec![Tensor::new(MetalBuffer(buffer))]
                            }
                        }
                    } else {
                        panic!("Tensor {} not found in file {}", sn, file_path.display());
                    }
                });

                // Check if this is a quantized weight
                let file = File::open(file_cache.get(file_name).unwrap()).unwrap();
                let buffer = unsafe { Mmap::map(&file).unwrap() };
                let tensors = SafeTensors::deserialize(&buffer).unwrap();
                if let Ok(tensor) = tensors.tensor(&safetensors_name) {
                    match tensor.dtype() {
                        safetensors::Dtype::F32
                        | safetensors::Dtype::F16
                        | safetensors::Dtype::BF16 => {}
                        _ => {
                            // This is a quantized weight
                            quantized_weights.push(node_index);
                        }
                    }
                }
            } else {
                println!(
                    "Warning: Tensor {} not found in model files",
                    safetensors_name
                );
            }
        }
    }

    quantized_weights
}
