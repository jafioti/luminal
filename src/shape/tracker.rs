use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 6]>,
    pub indexes: ArrayVec<[usize; 6]>,
    pub fake: ArrayVec<[bool; 6]>,
    pub mask: ArrayVec<[(Expression, Expression); 6]>,
    pub padding: ArrayVec<[(Expression, Expression); 6]>,
}

impl ShapeTracker {
    pub fn new(dims: &[Expression]) -> Self {
        let mut s = Self {
            dims: Default::default(),
            indexes: Default::default(),
            fake: Default::default(),
            mask: Default::default(),
            padding: Default::default(),
        };
        for (i, d) in dims.iter().enumerate() {
            s.dims.push(*d);
            s.indexes.push(i);
            s.fake.push(false);
            s.mask.push((0.into(), i32::MAX.into())); // Unset upper bound mask are i32::MAX
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
    pub fn add_dim(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.indexes.insert(axis, self.dims.len());
        self.dims.push(dim.into());
        self.fake.push(false);
        self.mask.push((0.into(), i32::MAX.into()));
        self.padding.push((0.into(), 0.into()));
    }

    /// Add fake dim along a certian axis
    pub fn expand(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.add_dim(axis, dim);
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
        self.mask.remove(index);
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

    /// Create an expression to translate logical indexes into physical indexes
    pub fn index_expression(&self) -> BigExpression {
        println!("ORIG: {:?}", self);
        let shape = combine_dims(*self);
        println!("Combined: {:?}", shape);
        let strides = shape.unordered_strides(); // Dimension strides in original order
        let mut ind_expr = BigExpression::from(0); // The final index expression
        let mut current_elem_size = BigExpression::from(1); // Keep track of the size of each element of the current dim (last dim elem size: 1)

        // For combined dims
        // divide by last dims (smallest) element size
        // mod by combined dim size
        // multiply by last dims (smallest) stride

        // Loop through all dims in reverse order
        for i in shape.indexes.into_iter().rev() {
            // Get logical dimension size with padding and mask
            let current_size = pad_mask_dim(shape.dims[i], shape.padding[i], shape.mask[i]);
            // Don't include fake dimensions in the index expression
            if !shape.fake[i] {
                let mut dim_ind = BigExpression::from('z');
                // Remove other dim components
                dim_ind /= current_elem_size.clone();
                // Get position in current dim
                dim_ind %= current_size.clone();
                // Add offset
                dim_ind += shape.mask[i].0 - shape.padding[i].0;
                // Multiply by stride
                dim_ind *= strides[i].clone();
                // Add to index expression
                ind_expr += dim_ind;
            }
            // Keep track of element size for next dimension
            current_elem_size *= current_size;
        }
        ind_expr.simplify()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> BigExpression {
        let shape = combine_dims(*self);
        let mut ret = BigExpression::from(1);
        let mut acc = BigExpression::from(1);
        let logical = BigExpression::from('z');
        for i in shape.indexes.into_iter().rev() {
            let (bottom_slice, top_slice) = shape.mask[i];
            let logical_sh = pad_mask_dim(shape.dims[i], shape.padding[i], shape.mask[i]);
            if !shape.fake[i] {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                let greater_than = shape.padding[i].0.big() - bottom_slice;
                if greater_than != 0 {
                    ret &= dim_ind.clone().gte(greater_than);
                }
                ret &= dim_ind.lt(shape.dims[i].big() + shape.padding[i].0);
                if top_slice
                    .to_usize()
                    .map(|s| shape.dims[i].to_usize().map(|dim| s < dim).unwrap_or(true))
                    .unwrap_or(true)
                {
                    ret = ret.min(top_slice);
                }
            }
            acc *= logical_sh;
        }
        ret.simplify()
    }

    /// The number of elements in this tensor, including padding and mask
    pub fn n_elements(&self) -> BigExpression {
        self.shape().into_iter().product::<BigExpression>().max(1)
    }

    /// The number of elements in this tensor, not including pads and mask
    pub fn n_physical_elements(&self) -> BigExpression {
        self.indexes
            .into_iter()
            .filter(|i| !self.fake[*i])
            .map(|i| self.dims[i].big())
            .product::<BigExpression>()
            .max(1)
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
        Self::new(
            &self
                .shape()
                .into_iter()
                .map(|i| i.small())
                .collect::<Vec<_>>(),
        )
    }

    /// Check if contiguous (no permutes or fake dimensions)
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
            .map(|i| pad_mask_dim(self.dims[i], self.padding[i], self.mask[i]))
            .collect()
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape().iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Take a slice
    pub fn slice(&mut self, mask: &[(Expression, Expression)]) {
        for (ind, (b, t)) in mask.iter().enumerate().map(|(i, m)| (self.indexes[i], m)) {
            self.mask[ind].0 = self.mask[ind].0.max(b.max(0));
            self.mask[ind].1 = self.mask[ind].1.min(t.max(0));
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(Expression, Expression)]) {
        for (ind, (s, e)) in padding
            .iter()
            .enumerate()
            .map(|(i, m)| (self.indexes[i], m))
        {
            // Make sure we aren't padding a masked dimension
            if (e.to_usize().map(|n| n != 0).unwrap_or(true)
                && self.mask[ind]
                    .1
                    .to_usize()
                    .map(|n| n as i32 != i32::MAX)
                    .unwrap_or(true))
                || (s.to_usize().map(|n| n != 0).unwrap_or(true)
                    && self.mask[ind]
                        .0
                        .to_usize()
                        .map(|n| n as i32 != 0)
                        .unwrap_or(true))
            {
                panic!("Adding padding to a masked shape isn't supported")
            }
            self.padding[ind].0 += s.max(0);
            self.padding[ind].1 += e.max(0);
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
        for (a, b) in self.mask.iter_mut() {
            *a = a.exec_stack(dyn_dim_map, stack).unwrap().into();
            *b = b.exec_stack(dyn_dim_map, stack).unwrap().into();
        }
    }

    pub fn is_sliced(&self) -> bool {
        self.mask.iter().any(|(b, e)| {
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

fn pad_mask_dim(
    dim: impl Into<BigExpression>,
    padding: (Expression, Expression),
    mask: (Expression, Expression),
) -> BigExpression {
    (dim.into() + padding.0 + padding.1).min(mask.1) - mask.0
}

// Combine non-permuted, non-padded, non-fake, non-masked dimensions together
fn combine_dims(mut shape: ShapeTracker) -> ShapeTracker {
    for i in (1..shape.len()).rev() {
        let (ind_i, ind_i_minus_1) = (shape.indexes[i], shape.indexes[i - 1]);
        // Test permute
        if (ind_i != ind_i_minus_1 + 1)
            // Fakes
            || (shape.fake[ind_i] || shape.fake[ind_i_minus_1])
            // Dim i padding
            || (shape.padding[ind_i].0 != 0 || shape.padding[ind_i].1 != 0)
            // Dim i mask
            || (shape.mask[ind_i].0 != 0 || shape.mask[ind_i].1 != i32::MAX)
            // Dim i - 1 padding
            || (shape.padding[ind_i_minus_1].0 != 0 || shape.padding[ind_i_minus_1].1 != 0)
            // Dim i - 1 mask
            || (shape.mask[ind_i_minus_1].0 != 0 || shape.mask[ind_i_minus_1].1 != i32::MAX)
        {
            continue;
        }
        // We can combine dimension i and i - 1
        let dim_i = shape.dims[ind_i];
        shape.dims[ind_i_minus_1] *= dim_i;
        shape.remove_dim(i);
    }
    shape
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
        tracker.permute(&[2, 0, 1]);
        println!("Shape: [10, 5, 3]");
        println!("Strides: {:?}", tracker.strides());
        println!("Ind: {:?}", tracker.index_expression());
        println!("Val: {:?}", tracker.valid_expression());
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
