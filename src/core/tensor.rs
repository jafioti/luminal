use crate::shape::ShapeTracker;

/// An entirely dynamic tensor with data
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: ShapeTracker,
}

impl Tensor {
    /// Get the real data as layed out by the shape tracker
    pub fn real_data(&self) -> Vec<f32> {
        let mut data = vec![0.; self.shape.shape().iter().product()];
        let idx = self.shape.index_fn();
        for (i, r) in data.iter_mut().enumerate() {
            *r = self.data[(idx)(i)];
        }

        data
    }
}
