use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 6]>,
    pub indexes: ArrayVec<[usize; 6]>,
    pub fake: ArrayVec<[bool; 6]>,
    pub slices: ArrayVec<[(Expression, Expression); 6]>,
    pub padding: ArrayVec<[(Expression, Expression); 6]>,
}

impl ShapeTracker {
    pub fn new(dims: &[Expression]) -> Self {
        let mut s = Self {
            dims: Default::default(),
            indexes: Default::default(),
            fake: Default::default(),
            slices: Default::default(),
            padding: Default::default(),
        };
        for (i, d) in dims.iter().enumerate() {
            s.dims.push(*d);
            s.indexes.push(i);
            s.fake.push(false);
            s.slices.push((0.into(), i32::MAX.into())); // Unset upper bound slices are i32::MAX
            s.padding.push((0.into(), 0.into()));
        }
        s
    }

    /// Create a shape tracker where all dims are fake
    pub fn fake(dims: &[Expression]) -> Self {
        let mut s = Self::new(dims);
        s.fake.iter_mut().for_each(|i| *i = true);
        s
    }

    /// Add dim along a certian axis
    pub fn add_dim(&mut self, axis: usize, dim: Expression) {
        self.indexes.insert(axis, self.dims.len());
        self.dims.push(dim);
        self.fake.push(false);
        self.slices.push((0.into(), i32::MAX.into()));
        self.padding.push((0.into(), 0.into()));
    }

    /// Add fake dim along a certian axis
    pub fn expand(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.add_dim(axis, dim.into());
        self.fake[self.indexes[axis]] = true;
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Expression {
        let index = self.indexes.remove(axis);
        self.fake.remove(index);
        for i in self.indexes.iter_mut() {
            if *i > index {
                *i -= 1;
            }
        }
        self.slices.remove(index);
        self.padding.remove(index);
        self.dims.remove(index)
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: &[usize]) {
        let new_indexes = axes.iter().map(|i| self.indexes[*i]).collect::<Vec<_>>();
        self.indexes.copy_from_slice(&new_indexes);
    }

    /// Strides without permute applied
    fn unordered_strides(&self) -> Vec<BigExpression> {
        let mut strides = (0..self.len())
            .rev()
            .scan(BigExpression::from(1), |state, i| {
                let ret = state.clone();
                if !self.fake[i] {
                    *state = state.clone() * self.dims[i];
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        strides
    }

    /// Compute strides
    pub fn strides(&self) -> Vec<BigExpression> {
        let strides = self.unordered_strides();
        self.indexes
            .into_iter()
            .map(|i| strides[i].clone())
            .collect()
    }

    pub fn index_expression(&self) -> BigExpression {
        // Create strides in original order
        let strides = self.unordered_strides();
        let mut ret = BigExpression::from(0);
        let mut acc = BigExpression::from(1);
        let logical = BigExpression::from('z');
        // Loop through all dims in reverse order
        for i in self.indexes.into_iter().rev() {
            let (start_padding, end_padding) = self.padding[i];
            let (start_slice, end_slice) = self.slices[i];
            let logical_sh =
                (self.dims[i].big() + start_padding + end_padding).min(end_slice) - start_slice;
            if !self.fake[i] {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                ret = ret
                    + (dim_ind - start_padding
                        + (start_slice.big() - start_padding.big().min(start_slice)))
                        * strides[i].clone();
            }
            acc = acc.clone() * logical_sh.clone();
        }
        ret.simplify()
    }

    /// If this BigExpression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> BigExpression {
        if !self.is_reshaped() {
            return 1.into();
        }
        let mut ret = BigExpression::from(1);
        let mut acc = BigExpression::from(1);
        let logical = BigExpression::from('z');
        for i in self.indexes.into_iter().rev() {
            let (bottom_padding, top_padding) = self.padding[i];
            let (bottom_slice, top_slice) = self.slices[i];
            let logical_sh =
                (self.dims[i].big() + bottom_padding + top_padding).min(top_slice) - bottom_slice;
            if !self.fake[i] {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                let greater_than = bottom_padding.big() - bottom_slice.big().min(bottom_padding);
                if greater_than != 0 {
                    ret = ret & dim_ind.clone().gte(greater_than);
                }
                ret = ret & dim_ind.lt(self.dims[i].big() + bottom_padding);
                if top_slice
                    .to_usize()
                    .map(|s| self.dims[i].to_usize().map(|dim| s < dim).unwrap_or(true))
                    .unwrap_or(true)
                {
                    ret = ret.min(top_slice);
                }
            }
            acc = acc * logical_sh;
        }
        ret.simplify()
    }

    /// The number of elements in this tensor, including pads and slices
    pub fn n_elements(&self) -> BigExpression {
        let r = self
            .indexes
            .into_iter()
            .map(|i| (i, self.dims[i].big()))
            // Add pads
            .map(|(i, dim)| (i, dim + self.padding[i].0 + self.padding[i].1))
            // Slice
            .map(|(i, dim)| dim.min(self.slices[i].1) - self.slices[i].0)
            .product();
        if r == 0 {
            1.into()
        } else {
            r
        }
    }

    /// The number of elements in this tensor, not including pads and slices
    pub fn n_physical_elements(&self) -> BigExpression {
        let r = self
            .dims
            .into_iter()
            // Filter out fake dimensions
            .enumerate()
            .filter(|(i, _)| !self.fake[*i])
            .map(|(_, i)| i.into())
            .product();
        if r == 0 {
            1.into()
        } else {
            r
        }
    }

    /// The number of dimensions
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn realize(mut self, dims: &[Expression]) -> Self {
        for (i, ind) in self.indexes.iter().enumerate() {
            self.dims[*ind] = dims[i];
        }
        self
    }

    /// Create a contiguous version
    pub fn contiguous(self) -> Self {
        let new_dims = self
            .indexes
            .into_iter()
            .map(|i| {
                self.dims[i].min(self.slices[i].1 - self.slices[i].0)
                    + self.padding[i].0
                    + self.padding[i].1
            })
            .collect::<Vec<_>>();
        Self::new(&new_dims)
    }

    /// Check if contiguous
    pub fn is_contiguous(&self) -> bool {
        self.indexes.iter().enumerate().all(|(a, b)| a == *b) && self.fake.iter().all(|i| !*i)
    }

    /// Check if this shape has been modified at all (permuted, sliced, or padded)
    pub fn is_reshaped(&self) -> bool {
        !self.is_contiguous() || self.is_sliced() || self.is_padded()
    }

    /// Realize the true shape
    pub fn shape(&self) -> Vec<BigExpression> {
        self.indexes
            .into_iter()
            .map(|i| {
                (self.dims[i].big() + self.padding[i].0 - self.slices[i].0 + self.padding[i].1)
                    .min(self.slices[i].1)
            })
            .collect()
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape().iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Take a slice
    pub fn slice(&mut self, slices: &[(Expression, Expression)]) {
        for (i, (s, e)) in slices.iter().enumerate() {
            self.slices[self.indexes[i]].0 = self.slices[self.indexes[i]].0.max(s.max(0));
            self.slices[self.indexes[i]].1 = self.slices[self.indexes[i]].1.min(e.max(0));
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(Expression, Expression)]) {
        for (i, (s, e)) in padding.iter().enumerate() {
            if (e.to_usize().map(|n| n != 0).unwrap_or(true)
                && self.slices[self.indexes[i]]
                    .1
                    .to_usize()
                    .map(|n| n as i32 != i32::MAX)
                    .unwrap_or(true))
                || (s.to_usize().map(|n| n != 0).unwrap_or(true)
                    && self.slices[self.indexes[i]]
                        .0
                        .to_usize()
                        .map(|n| n as i32 != 0)
                        .unwrap_or(true))
            {
                panic!("Adding padding to a slice isn't supported")
            }
            self.padding[self.indexes[i]].0 = self.padding[self.indexes[i]].0 + s.max(0);
            self.padding[self.indexes[i]].1 = self.padding[self.indexes[i]].1 + e.max(0);
        }
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_global_dyn_dims(&mut self, dyn_dim_map: &FxHashMap<char, usize>) {
        self.resolve_global_dyn_dims_stack(dyn_dim_map, &mut Vec::new());
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims. This function requires a stack to work with
    pub fn resolve_global_dyn_dims_stack(
        &mut self,
        dyn_dim_map: &FxHashMap<char, usize>,
        stack: &mut Vec<i32>,
    ) {
        for d in self.dims.iter_mut() {
            *d = d.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
        for (a, b) in self.padding.iter_mut() {
            *a = a.exec_stack(dyn_dim_map, stack).unwrap().into();
            *b = b.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
        for (a, b) in self.slices.iter_mut() {
            *a = a.exec_stack(dyn_dim_map, stack).unwrap().into();
            *b = b.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
    }

    pub fn is_sliced(&self) -> bool {
        self.slices.iter().any(|(b, e)| {
            b.to_usize().map(|i| i != 0).unwrap_or(true)
                || e.to_usize().map(|n| n as i32 != i32::MAX).unwrap_or(true)
        })
    }

    pub fn is_padded(&self) -> bool {
        self.padding.iter().any(|(b, e)| {
            b.to_usize().map(|i| i != 0).unwrap_or(true)
                || e.to_usize().map(|n| n != 0).unwrap_or(true)
        })
    }
}

/// Resolve shapes between the two trackers to the best of our ability
pub fn resolve_local_dyn_dims(a: &mut ShapeTracker, b: &mut ShapeTracker, default_to_one: bool) {
    // B to A
    for i in 0..a.dims.len() {
        if a.dims[a.indexes[i]].is_unknown() {
            a.dims[a.indexes[i]] = b.dims[b.indexes[i]];
            if a.dims[a.indexes[i]].is_unknown() && default_to_one {
                a.dims[a.indexes[i]] = 1.into();
            }
        }
    }

    // A to B
    for i in 0..a.dims.len() {
        if b.dims[b.indexes[i]].is_unknown() {
            b.dims[b.indexes[i]] = a.dims[a.indexes[i]];
            if b.dims[b.indexes[i]].is_unknown() && default_to_one {
                b.dims[b.indexes[i]] = 1.into();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_idx_expr() {
        let mut tracker = ShapeTracker::new(&[
            Expression::from(10),
            Expression::from(5),
            Expression::from(3),
        ]);
        tracker.permute(&[0, 2, 1]);
        println!("Strides: {:?}", tracker.strides());
        println!("Ind: {:?}", tracker.index_expression());
    }

    #[test]
    fn test_symbolic_idx() {
        let mut cx = Graph::new();
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        let a = cx.named_tensor::<R2<SEQ, HEAD_DIM>>("a").keep();
        let _b = cx.tensor::<R3<SEQ, HEAD_DIM_OVER_2, 1>>().keep();
        // Split input into evens and odds
        let split = a.reshape::<R3<SEQ, HEAD_DIM_OVER_2, 2>>();
        let x0: GraphTensor<R3<SEQ, HEAD_DIM_OVER_2, 1>> =
            split.slice((.., .., ..Expression::from(1))).realize();
        let _x1: GraphTensor<R3<SEQ, HEAD_DIM_OVER_2, 1>> =
            split.slice((.., .., Expression::from(1)..)).realize();

        println!("x0: {:?}", x0.shape.index_expression());
    }
}
