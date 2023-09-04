use tinyvec::ArrayVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Unknown,
}

impl Dim {
    pub fn to_usize(self) -> Option<usize> {
        match self {
            Dim::Known(n) => Some(n),
            Dim::Unknown => None,
        }
    }
}

impl Default for Dim {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Dim; 10]>,
    pub indexes: ArrayVec<[usize; 10]>,
    fake: ArrayVec<[bool; 10]>,
    slices: ArrayVec<[(usize, usize); 10]>,
    padding: ArrayVec<[(usize, usize); 10]>,
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
            s.slices.push((0, usize::MAX));
            s.padding.push((0, 0));
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
        self.slices.push((0, usize::MAX));
        self.padding.push((0, 0));
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) {
        let index = self.indexes.remove(axis);
        self.dims.remove(index);
        self.fake.remove(index);
        for i in self.indexes.iter_mut() {
            if *i > index {
                *i -= 1;
            }
        }
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: &[usize]) {
        self.indexes.copy_from_slice(axes);
    }

    /// Convert a logical index into a physical one
    pub fn index(&self, logical: usize) -> Option<usize> {
        let mut ret = 0;
        let mut acc = 1;
        let mut strides = self
            .dims
            .iter()
            .enumerate()
            .rev()
            .scan(1, |state, (i, x)| {
                let ret = *state;
                if !self.fake[i] {
                    *state *= x.to_usize().unwrap();
                }
                Some(ret)
            })
            .collect::<Vec<_>>();
        strides.reverse();
        for ind in self.indexes.iter().rev() {
            let sh = match self.dims[*ind] {
                Dim::Known(n) => n,
                Dim::Unknown => panic!("All dims must be known before indexing!"),
            };
            if !self.fake[*ind] {
                let dim_ind = (logical / acc) % sh + self.slices[*ind].0;
                ret += dim_ind * strides[*ind];
            }
            acc *= sh;
        }
        Some(ret)
    }

    /// The number of elements in this tensor. Counts unknown dims as size 0
    pub fn n_elements(&self) -> usize {
        self.dims
            .iter()
            .enumerate()
            .filter_map(|(ind, i)| {
                if let Dim::Known(n) = i {
                    Some((*n).min(self.slices[ind].1) - self.slices[ind].0)
                } else {
                    None
                }
            })
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
    pub fn contiguous(mut self) -> Self {
        self.indexes
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i);
        self
    }

    /// Check if contiguous
    pub fn is_contiguous(&self) -> bool {
        self.indexes.iter().enumerate().all(|(a, b)| a == *b)
    }

    /// Realize the true shape
    pub fn shape(&self) -> Vec<Dim> {
        let mut dims = Vec::with_capacity(self.dims.len());
        for i in self.indexes {
            dims.push(self.dims[i]);
        }
        dims
    }

    /// Take a slice
    pub fn slice(&mut self, slices: &[(usize, usize)]) {
        for (i, (s, e)) in slices.iter().enumerate() {
            // If we are slicing into a padded dim, remove as much padding as possible
            let mut s = *s;
            if self.padding[self.indexes[i]].0 != 0 {
                let padding = self.padding[self.indexes[i]].0;
                self.padding[self.indexes[i]].0 = self.padding[self.indexes[i]].0.saturating_sub(s);
                s = s.saturating_sub(padding);
            }
            self.slices[self.indexes[i]].0 += s;
            self.slices[self.indexes[i]].1 = self.slices[self.indexes[i]].1.min(*e);
        }
    }

    /// Add padding
    pub fn pad(&mut self, padding: &[(usize, usize)]) {
        for (i, (s, e)) in padding.iter().enumerate() {
            self.padding[self.indexes[i]].0 += *s;
            if *e != 0 && self.slices[self.indexes[i]].1 != 0 {
                panic!("Adding padding to a slice isn't supported")
            }
            self.padding[self.indexes[i]].1 += *e;
        }
    }
}

/// Resolve shapes between the two trackers to the best of our ability
pub fn resolve_shapes(a: &mut ShapeTracker, b: &mut ShapeTracker, default_to_one: bool) {
    // B to A
    for i in 0..a.dims.len() {
        if matches!(a.dims[a.indexes[i]], Dim::Unknown) {
            if let Dim::Known(n) = b.dims[b.indexes[i]] {
                a.dims[a.indexes[i]] = Dim::Known(n);
            } else if default_to_one {
                a.dims[a.indexes[i]] = Dim::Known(1);
            }
        }
    }

    // A to B
    for i in 0..a.dims.len() {
        if matches!(b.dims[b.indexes[i]], Dim::Unknown) {
            if let Dim::Known(n) = a.dims[a.indexes[i]] {
                b.dims[b.indexes[i]] = Dim::Known(n);
            } else if default_to_one {
                b.dims[b.indexes[i]] = Dim::Known(1);
            }
        }
    }
}
