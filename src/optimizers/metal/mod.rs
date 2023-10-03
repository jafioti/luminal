mod fp32;
use std::{
    any::{Any, TypeId},
    collections::{hash_map::DefaultHasher, HashSet},
    hash::{Hash, Hasher},
};

pub use fp32::*;
mod fp16;
pub use fp16::*;
use itertools::Itertools;
use metal_rs::*;
use regex::Regex;

use crate::prelude::{Data, Dim, ShapeTracker};

impl Data for Buffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    opts.set_preserve_invariance(true);
    let library = device.new_library_with_source(code, &opts).unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, None).unwrap()));

    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn hash<T: Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}

trait DispatchNElements {
    fn dispatch_n_elements(&self, n: usize);
}

impl DispatchNElements for ComputeCommandEncoderRef {
    fn dispatch_n_elements(&self, n: usize) {
        self.dispatch_thread_groups(
            MTLSize {
                width: (n as NSUInteger + 32) / 32,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
}

trait SetInt {
    fn set_int(&self, index: usize, value: u32);
}

impl SetInt for ComputeCommandEncoderRef {
    fn set_int(&self, index: usize, value: u32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
}

fn input_dyn_dims(
    shapes: &[(ShapeTracker, ShapeTracker)],
    encoder: &ComputeCommandEncoderRef,
    offset: usize,
) {
    let mut added = HashSet::new();
    for (d1, d2) in shapes
        .iter()
        .flat_map(|(a, b)| a.shape().into_iter().zip(b.shape()))
    {
        if let Dim::Unknown(c) = d1 {
            if !added.contains(&c) {
                encoder.set_int(offset + added.len(), d2.to_usize().unwrap() as u32);
                added.insert(c);
            }
        }
    }
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> String {
    shapes
        .iter()
        .flat_map(|st| st.shape())
        .filter_map(|d| {
            if let Dim::Unknown(c) = d {
                Some(c)
            } else {
                None
            }
        })
        .unique()
        .enumerate()
        .map(|(i, c)| format!(", device uint& {c} [[buffer({})]]", i + offset))
        .collect::<String>()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    let min_re = Regex::new(r"min\((.*?), (.*?)\)").unwrap();
    let max_re = Regex::new(r"max\((.*?), (.*?)\)").unwrap();
    let idx_exp = shape.index_expression().to_string();
    let idx_exp = max_re.replace_all(&idx_exp, "max((uint)($1), (uint)($2))");
    let idx_exp = min_re.replace_all(&idx_exp, "min((uint)($1), (uint)($2))");
    let val_exp = shape.valid_expression().to_string();
    let val_exp = max_re.replace_all(&val_exp, "max((uint)($1), (uint)($2))");
    let val_exp = min_re.replace_all(&val_exp, "min((uint)($1), (uint)($2))");
    (idx_exp.to_string(), val_exp.to_string())
}
