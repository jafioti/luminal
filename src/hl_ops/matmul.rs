use crate::{op::Function, prelude::*};

impl GraphTensor {
    pub fn matmul(mut self, mut rhs: GraphTensor) -> Self {
        if (self.shape.len() == 1 || self.shape.len() == 2) && rhs.shape.len() == 2 {
            let vec = self.shape.len() == 1;
            if vec {
                self = self.expand_dim(0, 1);
            }
            let (m, _) = self.dims2();
            let (_, n) = rhs.dims2();
            // Broadcasted Multiply
            let mul = self.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);

            // Sum Reduce
            let mut ret = mul.sum(2);
            if vec {
                ret = ret.reshape(ret.dims().last().unwrap());
            }
            ret
        } else if self.shape.len() == 2 && rhs.shape.len() == 1 {
            // Matrix x Vector: (m, n) x (n,) -> (m,)
            let (m, n) = self.dims2();
            let rhs_n = rhs.dims1();
            assert_eq!(n, rhs_n, "Matrix and vector dimensions must match");

            // Use a custom Function operator to avoid stack overflow from arithmetic operations
            let m_usize = m.to_usize().unwrap_or(0);
            let n_usize = n.to_usize().unwrap_or(0);
            let func = Function(
                "matrix_vector_matmul".to_string(),
                Box::new(move |inp| {
                    let (matrix_tensor, _) = &inp[0];
                    let (vector_tensor, _) = &inp[1];
                    let matrix_data = matrix_tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                    let vector_data = vector_tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();

                    let mut result = vec![0.0; m_usize];
                    for i in 0..m_usize {
                        for j in 0..n_usize {
                            result[i] += matrix_data[i * n_usize + j] * vector_data[j];
                        }
                    }
                    vec![Tensor::new(result)]
                }),
            );
            let final_id = self
                .graph()
                .add_op(func)
                .input(self.id, 0, self.shape)
                .input(rhs.id, 0, rhs.shape)
                .finish();
            let shape = ShapeTracker::new((m,));
            GraphTensor::from_id(final_id, shape, self.graph_ref)
        } else if self.shape.len() == 3 {
            let d = *rhs.dims().last().unwrap();
            let (a, b, _) = self.dims3();
            if rhs.shape.len() == 2 {
                // ABCxCD -> ABD
                // Reshape
                let w = rhs.permute((1, 0));

                // Broadcasted Multiply
                let mul = self.expand_dim(2, d) * w.expand_dim(0, a).expand_dim(1, b);

                // Sum Reduce
                mul.sum(3)
            } else if rhs.shape.len() == 3 {
                // Reshape
                let w = rhs.permute((0, 2, 1));

                // Broadcasted Multiply
                let mul = self.expand_dim(2, d) * w.expand_dim(1, b);

                // Sum Reduce
                mul.sum(3)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.shape.len() == 4 {
            let (a, b, c, _) = self.dims4();
            if rhs.shape.len() == 2 {
                // ABCDxDE -> ABCE
                let (_, e) = rhs.dims2();
                // Reshape
                rhs = rhs.permute((1, 0));
                // Broadcasted Multiply
                let mul =
                    self.expand_dim(3, e) * rhs.expand_dim(0, a).expand_dim(1, b).expand_dim(2, c);

                // Sum Reduce
                mul.sum(4)
            } else if rhs.shape.len() == 4 {
                // ABCDxABDE -> ABCE
                let (_, _, _, e) = rhs.dims4();
                // Reshape
                rhs = rhs.permute((0, 1, 3, 2));

                // Broadcasted Multiply
                let mul = self.expand_dim(3, e) * rhs.expand_dim(2, c);

                // Sum Reduce
                mul.sum(4)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.shape.len() == 5 && rhs.shape.len() == 5 {
            // ABCDExABCEF -> ABCDF
            let (a, b, c, e, f) = rhs.dims5();
            let (_, _, _, d, _) = self.dims5();
            // Reshape
            let w = rhs.reshape((a * b * c, e, f)).permute((0, 2, 1));
            let s = self.reshape((a * b * c, d, e));

            // Broadcasted Multiply
            let mul = s.expand_dim(2, f) * w.expand_dim(1, d);

            // Sum Reduce
            mul.sum(3).reshape((a, b, c, d, f))
        } else {
            panic!(
                "Can't matmul lhs {:?} and rhs {:?}",
                self.dims(),
                rhs.dims()
            )
        }
    }

    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor) -> GraphTensor {
        (self * rhs).sum(0)
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let b_data = random_vec(3);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = cx.tensor((3,)).set(b_data.clone());
        let c = a.matmul(b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<3>,));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let b_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = cx.tensor((3, 2)).set(b_data.clone());
        let c = a.matmul(b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<3>, DConst::<2>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_matmul() {
        let mut cx = Graph::new();
        let (a_data, b_data) = (random_vec(12), random_vec(8));
        let a = cx.tensor((2, 3, 2));
        a.set(a_data.clone());
        let b = cx.tensor((2, 4));
        b.set(b_data.clone());
        let c = a.matmul(b);
        c.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>, DConst::<2>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<2>, DConst::<4>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_batch_matmul() {
        let mut cx = Graph::new();
        let a_data = random_vec(12);
        let b_data = random_vec(12);
        let a = cx.tensor((2, 2, 3)).set(a_data.clone());
        let b = cx.tensor((2, 3, 2)).set(b_data.clone());
        let c = a.matmul(b).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>, DConst::<3>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<2>, DConst::<3>, DConst::<2>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_batch_matmul2() {
        let mut cx = Graph::new();
        let (a_data, b_data) = (random_vec(4), random_vec(6));
        let a = cx.tensor(('a', 'b'));
        a.set_dyn(a_data.clone(), (2, 2));
        let a = a.expand_dim(0, 1);
        let b = cx.tensor((1, 'b', 3));
        b.set_dyn(b_data.clone(), (1, 2, 3));
        let c = a.matmul(b);
        c.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<1>, DConst::<2>, DConst::<2>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<1>, DConst::<2>, DConst::<3>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }
}
