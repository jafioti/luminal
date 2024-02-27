mod matmul;
mod prim;

#[cfg(test)]
mod tests;

use cudarc::driver::{CudaSlice, DeviceRepr};
use itertools::Itertools;

use std::fmt::Write;

use luminal::prelude::*;

use self::symbolic::{BigExpression, Term};

pub type CudaCompiler<T> = (
    prim::CudaPrimitiveCompiler<T>,
    matmul::CudaMatMulCompiler<T>,
    prim::CopyCompiler<T>,
);

pub trait CudaFloat:
    std::fmt::Debug
    + Copy
    + cudarc::driver::DeviceRepr
    + std::marker::Unpin
    + cudarc::driver::ValidAsZeroBits
{
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

impl CudaFloat for f32 {
    fn from_f32(a: f32) -> Self {
        a
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn is_f32() -> bool {
        true
    }
    fn type_name() -> &'static str {
        "float"
    }
}
#[derive(Debug)]
pub struct CudaData<T>(CudaSlice<T>);

impl<T: DeviceRepr> Clone for CudaData<T> {
    fn clone(&self) -> Self {
        Self(self.0.try_clone().unwrap())
    }
}

impl Data for CudaData<f32> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl CudaFloat for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn is_f32() -> bool {
        false
    }
    fn type_name() -> &'static str {
        "__half"
    }
}
impl Data for CudaData<f16> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn expr_to_cuda_string(expr: BigExpression) -> String {
    let mut symbols = vec![];
    for term in expr.terms {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if c == 'z' {
                    "(int)idx".to_string()
                } else {
                    c.to_string()
                }
            }
            Term::Max => format!(
                "max((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Min => format!(
                "min((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            _ => format!(
                "({}{term:?}{})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
        };
        symbols.push(new_symbol);
    }
    symbols.pop().unwrap()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    (
        expr_to_cuda_string(shape.index_expression()),
        expr_to_cuda_string(shape.valid_expression()),
    )
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker]) -> (Vec<char>, String) {
    let symbols: Vec<char> = shapes
        .iter()
        .flat_map(|st| {
            st.shape()
                .into_iter()
                .chain(
                    st.padding
                        .into_iter()
                        .flat_map(|i| [i.0.into(), i.1.into()]),
                )
                .chain(st.slices.into_iter().flat_map(|i| [i.0.into(), i.1.into()]))
        })
        .flat_map(|d| d.to_symbols())
        .unique()
        .collect();
    (
        symbols.clone(),
        symbols.into_iter().fold(String::default(), |mut acc, c| {
            write!(&mut acc, ", const int {c}").unwrap();
            acc
        }),
    )
}
