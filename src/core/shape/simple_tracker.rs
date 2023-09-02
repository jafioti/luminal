use std::marker::PhantomData;

use petgraph::stable_graph::NodeIndex;
use tinyvec::ArrayVec;

use crate::prelude::Graph;

use super::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Unknown,
}

impl Default for Dim {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeTracker {
    dims: ArrayVec<[Dim; 10]>,
    indexes: ArrayVec<[usize; 10]>,
    fake: ArrayVec<[bool; 10]>,
}

impl ShapeTracker {
    pub fn new(dims: &[Dim]) -> Self {
        let mut s = Self {
            dims: Default::default(),
            indexes: Default::default(),
            fake: Default::default(),
        };
        #[allow(clippy::needless_range_loop)]
        for i in 0..dims.len() {
            s.dims.push(dims[i]);
            s.indexes.push(i);
            s.fake.push(false);
        }
        s
    }

    pub fn expand(&mut self, dim: Dim, axis: usize) {
        self.dims.push(dim);
        self.indexes.insert(axis, self.dims.len() - 1);
        self.fake.push(true);
    }

    pub fn remove_dim(&mut self, axis: usize) {
        let index = self.indexes.remove(axis);
        self.dims.remove(index);
        self.fake.remove(index);
    }

    pub fn permute(&mut self, axes: &[usize]) {
        self.indexes.copy_from_slice(axes);
    }

    pub fn index(&self, logical: usize) -> Option<usize> {
        let mut ret = 0;
        let mut acc = 1;
        for ind in self.indexes.iter().rev() {
            let sh = match self.dims[*ind] {
                Dim::Known(n) => n,
                Dim::Unknown => panic!("All dims must be known before indexing!"),
            };
            if !self.fake[*ind] {
                ret += ((logical / acc) % (sh)) * acc;
            }
            acc *= sh;
        }
        Some(ret)
    }

    pub fn n_elements(&self) -> usize {
        self.dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.fake[*i])
            .filter_map(|(_, i)| if let Dim::Known(n) = i { Some(n) } else { None })
            .product()
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn contiguous(mut self) -> Self {
        self.indexes
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i);
        self
    }
}

#[derive(Clone, Copy)]
pub struct GraphTensor<S: Shape> {
    pub id: NodeIndex,
    pub graph_ref: *mut Graph,
    pub(crate) _phantom: PhantomData<S>,
    pub shape: ShapeTracker,
}

// #[cfg(test)]
// mod tests {
//     use super::ShapeTracker;

//     #[test]
//     fn test_shape_tracker() {
//         let start = ShapeTracker::new(vec![Dim::Unknown, Dim::Unknown, Dim::Known(128)]); // A shape of batch x seq x embed
//                                                                                           // Strides: ([Unk[0] * Unk[1] * 128, Unk[1] * 128, 1])
//         start = start.permute([0, 2, 1]);
//         // Strides: ([Unk[0] * Unk[1] * 128, 1, Unk[1] * 128])
//         start = start.reshape() // Non-contiguous, so we need a contiguous call first
//     }
// }
