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

#[derive(Debug, Clone, Copy)]
pub struct ShapeTracker {
    dims: ArrayVec<[Dim; 10]>,
    indexes: ArrayVec<[usize; 10]>,
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
        #[allow(clippy::needless_range_loop)]
        for i in 0..dims.len() {
            s.dims.push(dims[i]);
            s.indexes.push(i);
            s.fake.push(false);
            s.slices.push((0, usize::MAX));
            s.padding.push((0, 0));
        }
        s
    }

    /// Add dim along a certian axis
    pub fn expand(&mut self, axis: usize, dim: Dim) {
        self.dims.push(dim);
        self.indexes.insert(axis, self.dims.len() - 1);
        self.fake.push(true);
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) {
        let index = self.indexes.remove(axis);
        self.dims.remove(index);
        self.fake.remove(index);
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: &[usize]) {
        self.indexes.copy_from_slice(axes);
    }

    /// Convert a logical index into a physical one
    pub fn index(&self, logical: usize) -> Option<usize> {
        let mut ret = 0;
        let mut acc = 1;
        for ind in self.indexes.iter().rev() {
            let sh = match self.dims[*ind] {
                Dim::Known(n) => n,
                Dim::Unknown => panic!("All dims must be known before indexing!"),
            };
            if !self.fake[*ind] {
                let dim_ind = (logical / acc) % (sh) + self.slices[*ind].0;
                if dim_ind < self.padding[*ind].0 || dim_ind > (sh - self.padding[*ind].1) {
                    return None;
                }
                ret += dim_ind * acc;
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
            .filter(|(i, _)| !self.fake[*i])
            .filter_map(|(_, i)| if let Dim::Known(n) = i { Some(n) } else { None })
            .product()
    }

    /// The number of dimensions
    pub fn len(&self) -> usize {
        self.dims.len()
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
                self.padding[self.indexes[i]].0.saturating_sub(s);
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
            self.padding[self.indexes[i]].1 += *e;
        }
    }
}
