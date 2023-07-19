use crate::shape::ShapeTracker;

/// An entirely dynamic tensor with data
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: ShapeTracker,
}
