mod fp16;
mod fp32;

pub use fp16::CudaFp16Compiler;
pub use fp32::CudaFp32Compiler;
use half::f16;

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::{op::*, prelude::*};

/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct CudaCopyToDevice<T>(Arc<CudaDevice>, PhantomData<T>);

impl<T> CudaCopyToDevice<T> {
    fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyToDevice(dev, Default::default())
    }
}

impl<T> PartialEq for CudaCopyToDevice<T> {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl<T> Debug for CudaCopyToDevice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaCopyToDevice")
    }
}

impl<T> Operator for CudaCopyToDevice<T>
where
    CudaSlice<T>: Data,
    T: ConvertF32 + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<CudaSlice<T>>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cpu_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<Vec<f32>>()
            .unwrap();
        let vec = cpu_data
            .iter()
            .copied()
            .map(ConvertF32::from)
            .collect::<Vec<_>>();
        let mut a = unsafe { self.0.alloc::<T>(vec.len()).unwrap() };
        self.0.htod_copy_into(vec, &mut a).unwrap();
        vec![Tensor { data: Box::new(a) }]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone)]
pub struct CudaCopyFromDevice<T>(Arc<CudaDevice>, PhantomData<T>);
impl<T> PartialEq for CudaCopyFromDevice<T> {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl<T> CudaCopyFromDevice<T> {
    fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyFromDevice(dev, Default::default())
    }
}

impl<T> Debug for CudaCopyFromDevice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaCopyToDevice")
    }
}

impl<T> Operator for CudaCopyFromDevice<T>
where
    CudaSlice<T>: Data,
    T: ConvertF32 + cudarc::driver::DeviceRepr + std::marker::Unpin,
{
    fn process(&self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().data.as_any().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cuda_data = inp[0]
            .0
            .borrowed()
            .data
            .as_any()
            .downcast_ref::<CudaSlice<T>>()
            .unwrap();
        vec![Tensor {
            data: Box::new(
                self.0
                    .dtoh_sync_copy(cuda_data)
                    .unwrap()
                    .into_iter()
                    .map(ConvertF32::to)
                    .collect::<Vec<_>>(),
            ),
        }]
    }
}

pub trait ConvertF32 {
    fn to(self) -> f32;
    fn from(a: f32) -> Self;
}

impl ConvertF32 for f32 {
    fn from(a: f32) -> Self {
        a
    }
    fn to(self) -> f32 {
        self
    }
}

impl ConvertF32 for f16 {
    fn from(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to(self) -> f32 {
        self.to_f32()
    }
}
