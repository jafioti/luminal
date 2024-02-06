use std::{borrow::Cow, collections::HashMap, fs::File};

use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, View};

fn main() {
    println!("Converting Mistral weights");
    let root_path = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let file_paths = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ];
    for file_path in file_paths.iter() {
        let mut weights = HashMap::new();
        let file = File::open(format!("{root_path}/../mistral-7b-hf/{file_path}")).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let st = SafeTensors::deserialize(&buffer).unwrap();

        for (weight_name, tensor) in st.tensors() {
            let bytes = tensor.data().to_vec();
            let mut data: Vec<f16> = match tensor.dtype() {
                Dtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
                Dtype::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f16::from_f32(f32::from_ne_bytes([c[0], c[1], c[2], c[3]])))
                    .collect(),
                Dtype::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_f32(bf16::from_ne_bytes([c[0], c[1]]).to_f32()))
                    .collect(),
                _ => panic!("{:?} is not a supported dtype", tensor.dtype()),
            };
            println!("Converted {weight_name}");
            let len = data.len();
            weights.insert(weight_name, Fp16Vec(data, vec![len]));
        }
        safetensors::serialize_to_file(
            weights,
            &None,
            format!("{root_path}/../mistral-7b-hf/converted-{file_path}").as_ref(),
        )
        .unwrap();
    }
}

struct Fp16Vec(Vec<f16>, Vec<usize>);

impl View for Fp16Vec {
    fn dtype(&self) -> Dtype {
        Dtype::F16
    }

    fn shape(&self) -> &[usize] {
        &self.1
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        let float_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                self.0.len() * std::mem::size_of::<f16>(),
            )
        };
        Cow::Borrowed(float_slice)
    }

    fn data_len(&self) -> usize {
        self.0.len() * std::mem::size_of::<f16>()
    }
}
