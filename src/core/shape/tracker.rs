use std::collections::HashMap;

use tinyvec::ArrayVec;

use super::symbolic::{BigExpression, Expression};

#[derive(Debug, Clone, Copy, PartialEq)]
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
        for i in 0..dims.len() {
            s.fake[i] = true;
        }
        s
    }

    /// Add dim along a certian axis
    pub fn expand(&mut self, axis: usize, dim: Expression) {
        self.indexes.insert(axis, self.dims.len());
        self.dims.push(dim);
        self.fake.push(true);
        self.slices.push((0.into(), i32::MAX.into()));
        self.padding.push((0.into(), 0.into()));
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
    fn unordered_strides(&self) -> Vec<Expression> {
        let mut strides = self
            .dims
            .iter()
            .enumerate()
            .rev()
            .scan(Expression::from(1), |state, (i, x)| {
                let ret = *state;
                if !self.fake[i] {
                    *state = *state * *x;
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        strides
    }

    /// Compute strides
    pub fn strides(&self) -> Vec<Expression> {
        let strides = self.unordered_strides();
        self.indexes.into_iter().map(|i| strides[i]).collect()
    }

    pub fn index_expression(&self) -> BigExpression {
        // Create strides in original order
        let mut strides = self
            .dims
            .iter()
            .enumerate()
            .rev()
            .scan(BigExpression::from(1), |state, (i, x)| {
                let ret = state.clone();
                if !self.fake[i] {
                    *state = state.clone() * *x;
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        let mut ret = BigExpression::from(0);
        let mut acc = BigExpression::from(1);
        let logical = BigExpression::from('z');
        // Loop through all dims in current order
        for (sh, stride, padding, slice, fake) in self.indexes.into_iter().rev().map(|i| {
            (
                self.dims[i],
                strides[i].clone(),
                self.padding[i],
                self.slices[i],
                self.fake[i],
            )
        }) {
            let logical_sh =
                (BigExpression::from(sh) + padding.0 + padding.1).min(slice.1) - slice.0;
            if !fake {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                ret = ret
                    + (dim_ind - padding.0
                        + (BigExpression::from(slice.0)
                            - BigExpression::from(padding.0).min(slice.0)))
                        * stride;
            }
            acc = acc.clone() * logical_sh.clone();
        }
        ret.minimize()
    }

    /// If this BigExpression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> BigExpression {
        let mut ret = BigExpression::from(1);
        let mut acc = BigExpression::from(1);
        let logical = BigExpression::from('z');
        for (sh, padding, slice, fake) in self
            .indexes
            .into_iter()
            .rev()
            .map(|i| (self.dims[i], self.padding[i], self.slices[i], self.fake[i]))
        {
            let logical_sh =
                (BigExpression::from(sh) + padding.0 + padding.1).min(slice.1) - slice.0;
            if !fake {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                ret = ret
                    & dim_ind.clone().gte(
                        BigExpression::from(padding.0)
                            - BigExpression::from(slice.0).min(padding.0),
                    );
                ret = ret & dim_ind.lt((BigExpression::from(sh) + padding.0).min(slice.1));
            }
            acc = acc * logical_sh;
        }
        ret.minimize()
    }

    /// The number of elements in this tensor, including pads and slices. Counts unknown dims as size 0
    pub fn n_elements(&self) -> usize {
        self.indexes
            .into_iter()
            // Filter out unknowns
            .filter_map(|i| self.dims[i].to_usize().map(|n| (i, n)))
            // Add pads
            .map(|(ind, dim)| {
                (
                    ind,
                    dim + self.padding[ind].0.to_usize().unwrap()
                        + self.padding[ind].1.to_usize().unwrap(),
                )
            })
            // Slice
            .map(|(ind, dim)| {
                dim.min(self.slices[ind].1.to_usize().unwrap_or(i32::MAX as usize))
                    - self.slices[ind].0.to_usize().unwrap_or_default()
            })
            .product()
    }

    /// The number of elements in this tensor, not including pads and slices. Counts unknown dims as size 0
    pub fn n_physical_elements(&self) -> usize {
        self.dims
            .into_iter()
            // Filter out fake dimensions
            .enumerate()
            .filter(|(i, _)| !self.fake[*i])
            .flat_map(|(_, i)| i.to_usize())
            .product()
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
        self.indexes.iter().enumerate().all(|(a, b)| a == *b)
    }

    /// Realize the true shape
    pub fn shape(&self) -> Vec<BigExpression> {
        self.indexes
            .into_iter()
            .map(|i| {
                (BigExpression::from(self.dims[i]) + self.padding[i].0 + self.padding[i].1)
                    .min(self.slices[i].1)
                    - self.slices[i].0
            })
            .collect()
    }

    /// Take a slice
    pub fn slice(&mut self, slices: &[(Expression, Expression)]) {
        for (i, (s, e)) in slices.iter().enumerate() {
            self.slices[self.indexes[i]].0 = self.slices[self.indexes[i]].0.max(*s);
            self.slices[self.indexes[i]].1 = self.slices[self.indexes[i]].1.min(*e);
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(Expression, Expression)]) {
        for (i, (s, e)) in padding.iter().enumerate() {
            if e.to_usize().map(|n| n != 0).unwrap_or(true)
                && self.slices[self.indexes[i]]
                    .1
                    .to_usize()
                    .map(|n| n as i32 != i32::MAX)
                    .unwrap_or(true)
            {
                panic!("Adding padding to a slice isn't supported")
            }
            self.padding[self.indexes[i]].0 = self.padding[self.indexes[i]].0 + *s;
            self.padding[self.indexes[i]].1 = self.padding[self.indexes[i]].1 + *e;
        }
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_global_dyn_dims(mut self, dyn_dim_map: &HashMap<char, usize>) -> Self {
        for d in self.dims.iter_mut() {
            *d = d.exec(dyn_dim_map).unwrap().into();
        }
        for (a, b) in self.padding.iter_mut() {
            *a = a.exec(dyn_dim_map).unwrap().into();
            *b = b.exec(dyn_dim_map).unwrap().into();
        }
        for (a, b) in self.slices.iter_mut() {
            *a = a.exec(dyn_dim_map).unwrap().into();
            *b = b.exec(dyn_dim_map).unwrap().into();
        }
        self
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
