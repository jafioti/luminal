use std::collections::HashMap;

use tinyvec::ArrayVec;

use super::symbolic::{BigExprInterface, BigExpression};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Unknown(char),
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Dim::Known(value)
    }
}
impl From<&usize> for Dim {
    fn from(value: &usize) -> Self {
        Dim::Known(*value)
    }
}

impl Dim {
    pub fn to_usize(self) -> Option<usize> {
        match self {
            Dim::Known(n) => Some(n),
            Dim::Unknown(_) => None,
        }
    }
}

impl Default for Dim {
    fn default() -> Self {
        Self::Unknown('-')
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Dim; 6]>,
    pub indexes: ArrayVec<[usize; 6]>,
    pub fake: ArrayVec<[bool; 6]>,
    pub slices: ArrayVec<[(Dim, Dim); 6]>,
    pub padding: ArrayVec<[(Dim, Dim); 6]>,
}

impl ShapeTracker {
    pub fn new(dims: &[Dim]) -> Self {
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
            s.slices
                .push((Dim::Known(0), Dim::Known(i32::MAX as usize))); // Unset upper bound slices are i32::MAX
            s.padding.push((Dim::Known(0), Dim::Known(0)));
        }
        s
    }

    /// Create a shape tracker where all dims are fake
    pub fn fake(dims: &[Dim]) -> Self {
        let mut s = Self::new(dims);
        for i in 0..dims.len() {
            s.fake[i] = true;
        }
        s
    }

    /// Add dim along a certian axis
    pub fn expand(&mut self, axis: usize, dim: Dim) {
        self.indexes.insert(axis, self.dims.len());
        self.dims.push(dim);
        self.fake.push(true);
        self.slices
            .push((Dim::Known(0), Dim::Known(i32::MAX as usize)));
        self.padding.push((Dim::Known(0), Dim::Known(0)));
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Dim {
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
    fn unordered_strides(&self) -> Vec<usize> {
        let mut strides = self
            .dims
            .iter()
            .enumerate()
            .rev()
            .scan(1, |state, (i, x)| {
                let ret = *state;
                if !self.fake[i] {
                    *state *= x
                        .to_usize()
                        .expect("All dims must be known before indexing!");
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        strides
    }

    /// Once all dims are known, compute strides
    pub fn strides(&self) -> Vec<usize> {
        let strides = self.unordered_strides();
        self.indexes.into_iter().map(|i| strides[i]).collect()
    }

    /// Create an indexer to convert logical indexes into physical indexes
    pub fn indexer(&self) -> Indexer {
        let strides = self.unordered_strides();
        Indexer {
            data: self
                .indexes
                .into_iter()
                .rev()
                .map(|i| {
                    (
                        self.dims[i].to_usize().unwrap(),
                        strides[i],
                        (
                            self.padding[i].0.to_usize().unwrap(),
                            self.padding[i].1.to_usize().unwrap(),
                        ),
                        (
                            self.slices[i].0.to_usize().unwrap(),
                            self.slices[i].1.to_usize().unwrap(),
                        ),
                        self.fake[i],
                    )
                })
                .collect(),
        }
    }

    pub fn index_expression(&self) -> BigExpression {
        let mut strides = self
            .dims
            .iter()
            .enumerate()
            .rev()
            .scan(1.big_expr(), |state, (i, x)| {
                let ret = state.clone();
                if !self.fake[i] {
                    *state = state.clone() * dim_to_expression(*x);
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        let mut ret = 0.big_expr();
        let mut acc = 1.big_expr();
        let logical = 'z'.big_expr();
        for (sh, stride, padding, slice, fake) in self.indexes.into_iter().rev().map(|i| {
            (
                dim_to_expression(self.dims[i]),
                strides[i].clone(),
                self.padding[i],
                self.slices[i],
                self.fake[i],
            )
        }) {
            let logical_sh = (sh + dim_to_expression(padding.0) + dim_to_expression(padding.1))
                .min(dim_to_expression(slice.1))
                - dim_to_expression(slice.0);
            if !fake {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                ret = ret
                    + (dim_ind - dim_to_expression(padding.0)
                        + (dim_to_expression(slice.0)
                            - dim_to_expression(padding.0).min(dim_to_expression(slice.0))))
                        * stride;
            }
            acc = acc.clone() * logical_sh.clone();
        }
        ret
    }

    /// If this BigExpression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> BigExpression {
        let mut ret = 1.big_expr();
        let mut acc = 1.big_expr();
        let logical = 'z'.big_expr();
        for (sh, padding, slice, fake) in self.indexes.into_iter().rev().map(|i| {
            (
                dim_to_expression(self.dims[i]),
                self.padding[i],
                self.slices[i],
                self.fake[i],
            )
        }) {
            let logical_sh =
                (sh.clone() + dim_to_expression(padding.0) + dim_to_expression(padding.1))
                    .min(dim_to_expression(slice.1))
                    - dim_to_expression(slice.0);
            if !fake {
                let dim_ind = (logical.clone() / acc.clone()) % logical_sh.clone();
                ret = ret
                    & dim_ind.clone().gte(
                        dim_to_expression(padding.0)
                            - dim_to_expression(slice.0).min(dim_to_expression(padding.0)),
                    );
                ret = ret
                    & dim_ind
                        .lt((sh + dim_to_expression(padding.0)).min(dim_to_expression(slice.1)));
            }
            acc = acc * logical_sh;
        }
        ret
    }

    /// The number of elements in this tensor, including pads and slices. Counts unknown dims as size 0
    pub fn n_elements(&self) -> usize {
        self.indexes
            .into_iter()
            // Filter out unknowns
            .filter_map(|i| {
                if let Dim::Known(n) = self.dims[i] {
                    Some((i, n))
                } else {
                    None
                }
            })
            // Add pads
            .map(|(ind, dim)| {
                (
                    ind,
                    // dim + self.padding[ind].0.to_usize().unwrap_or_default()
                    //     + self.padding[ind].1.to_usize().unwrap_or_default(),
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

    pub fn realize(mut self, dims: &[Dim]) -> Self {
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
            .map(|i| match self.dims[i] {
                Dim::Known(n) => Dim::Known(
                    n.min(
                        self.slices[i].1.to_usize().unwrap_or(i32::MAX as usize)
                            - self.slices[i].0.to_usize().unwrap_or_default(),
                    ) + self.padding[i].0.to_usize().unwrap_or_default()
                        + self.padding[i].1.to_usize().unwrap_or_default(),
                ),
                Dim::Unknown(c) => Dim::Unknown(c),
            })
            .collect::<Vec<_>>();
        Self::new(&new_dims)
    }

    /// Check if contiguous
    pub fn is_contiguous(&self) -> bool {
        self.indexes.iter().enumerate().all(|(a, b)| a == *b)
    }

    /// Realize the true shape
    pub fn shape(&self) -> Vec<Dim> {
        self.indexes.into_iter().map(|i| self.dims[i]).collect()
    }

    /// Take a slice
    pub fn slice(&mut self, slices: &[(Dim, Dim)]) {
        for (i, (s, e)) in slices.iter().enumerate() {
            // If we are slicing into a padded dim, remove as much padding as possible
            if let Dim::Known(mut s) = s {
                if self.padding[self.indexes[i]]
                    .0
                    .to_usize()
                    .map(|n| n != 0)
                    .unwrap_or_default()
                {
                    let padding = self.padding[self.indexes[i]].0.to_usize().unwrap();
                    self.padding[self.indexes[i]].0 = Dim::Known(padding.saturating_sub(s));
                    s = s.saturating_sub(padding);
                }
                if let Dim::Known(a) = self.slices[self.indexes[i]].0 {
                    self.slices[self.indexes[i]].0 = Dim::Known(a + s);
                } else {
                    self.slices[self.indexes[i]].0 = Dim::Known(s);
                }
            } else {
                self.slices[self.indexes[i]].0 = *s;
            }
            if let Dim::Known(a) = self.slices[self.indexes[i]].1 {
                if let Dim::Known(e) = e {
                    self.slices[self.indexes[i]].1 = Dim::Known(a.min(*e));
                } else {
                    self.slices[self.indexes[i]].1 = *e;
                }
            } else {
                self.slices[self.indexes[i]].1 = *e;
            }
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(Dim, Dim)]) {
        for (i, (s, e)) in padding.iter().enumerate() {
            if let Dim::Known(a) = self.padding[self.indexes[i]].0 {
                if let Dim::Known(s) = s {
                    self.padding[self.indexes[i]].0 = Dim::Known(a + *s);
                } else {
                    self.padding[self.indexes[i]].0 = *s;
                }
            } else {
                self.padding[self.indexes[i]].0 = *s;
            }
            if e.to_usize().map(|n| n != 0).unwrap_or_default()
                && self.slices[self.indexes[i]]
                    .1
                    .to_usize()
                    .map(|n| n as i32 != i32::MAX)
                    .unwrap_or_default()
            {
                panic!("Adding padding to a slice isn't supported")
            }
            if let Dim::Known(a) = self.padding[self.indexes[i]].1 {
                if let Dim::Known(e) = e {
                    self.padding[self.indexes[i]].1 = Dim::Known(a + *e);
                } else {
                    self.padding[self.indexes[i]].1 = *e;
                }
            } else {
                self.padding[self.indexes[i]].1 = *e;
            }
        }
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_global_dyn_dims(mut self, dyn_dim_map: &HashMap<char, usize>) -> Self {
        for d in self.dims.iter_mut() {
            if let Dim::Unknown(u) = *d {
                *d = Dim::Known(dyn_dim_map[&u]);
            }
        }
        for (a, b) in self.padding.iter_mut() {
            if let Dim::Unknown(u) = *a {
                *a = Dim::Known(dyn_dim_map[&u]);
            }
            if let Dim::Unknown(u) = *b {
                *b = Dim::Known(dyn_dim_map[&u]);
            }
        }
        for (a, b) in self.slices.iter_mut() {
            if let Dim::Unknown(u) = *a {
                *a = Dim::Known(dyn_dim_map[&u]);
            }
            if let Dim::Unknown(u) = *b {
                *b = Dim::Known(dyn_dim_map[&u]);
            }
        }
        self
    }

    pub fn is_sliced(&self) -> bool {
        self.slices
            .iter()
            .any(|(b, e)| if let Dim::Known(n) = b {*n != 0} else {true} || if let Dim::Known(n) = e {*n as i32 != i32::MAX} else {false})
    }

    pub fn is_padded(&self) -> bool {
        self.padding
            .iter()
            .any(|(b, e)| if let Dim::Known(n) = b {*n != 0} else {true} || if let Dim::Known(n) = e {*n != 0} else {true})
    }
}

/// Resolve shapes between the two trackers to the best of our ability
pub fn resolve_local_dyn_dims(a: &mut ShapeTracker, b: &mut ShapeTracker, default_to_one: bool) {
    // B to A
    for i in 0..a.dims.len() {
        if matches!(a.dims[a.indexes[i]], Dim::Unknown('-')) {
            a.dims[a.indexes[i]] = b.dims[b.indexes[i]];
            if matches!(a.dims[a.indexes[i]], Dim::Unknown('-')) && default_to_one {
                a.dims[a.indexes[i]] = Dim::Known(1);
            }
        }
    }

    // A to B
    for i in 0..a.dims.len() {
        if matches!(b.dims[b.indexes[i]], Dim::Unknown('-')) {
            b.dims[b.indexes[i]] = a.dims[a.indexes[i]];
            if matches!(b.dims[b.indexes[i]], Dim::Unknown('-')) && default_to_one {
                b.dims[b.indexes[i]] = Dim::Known(1);
            }
        }
    }
}

pub struct Indexer {
    #[allow(clippy::type_complexity)]
    data: ArrayVec<[(usize, usize, (usize, usize), (usize, usize), bool); 6]>,
}

impl Indexer {
    /// Convert a logical index into a physical index
    pub fn index(&self, logical: usize) -> Option<usize> {
        let mut ret = 0;
        let mut acc = 1;
        for (sh, stride, padding, slice, fake) in self.data.into_iter() {
            let logical_sh = (sh + padding.0 + padding.1).min(slice.1) - slice.0;
            if !fake {
                let dim_ind = (logical / acc) % logical_sh;
                // Over top or under bottom
                if dim_ind >= (sh + padding.0).min(slice.1)
                    || dim_ind < padding.0.saturating_sub(slice.0)
                {
                    return None;
                }
                ret += (dim_ind - padding.0 + (slice.0.saturating_sub(padding.0))) * stride;
            }
            acc *= logical_sh;
        }
        Some(ret)
    }
}

fn dim_to_expression(d: Dim) -> BigExpression {
    match d {
        Dim::Known(n) => n.big_expr(),
        Dim::Unknown(c) => c.big_expr(),
    }
}
