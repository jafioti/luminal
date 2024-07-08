use crate::prelude::*;

impl GraphTensor {
    pub fn matmul(mut self, rhs: GraphTensor) -> Self {
        if (self.shape.len() == 1 || self.shape.len() == 2) && rhs.shape.len() == 2 {
            let vec = self.shape.len() == 1;
            if vec {
                self = self.expand(0, 1);
            }
            let m = self.shape()[0].small();
            let n = rhs.shape()[1].small();
            // Broadcasted Multiply
            let mul = self.expand(1, n) * rhs.permute((1, 0)).expand(0, m);

            // Sum Reduce
            let mut ret = mul.sum_reduce(2);
            if vec {
                ret = ret.reshape(ret.shape().last().unwrap());
            }
            ret
        } else if self.shape.len() == 3 {
            let d = rhs.shape().last().unwrap().small();
            let a = self.shape()[0].small();
            let b = self.shape()[1].small();
            if rhs.shape.len() == 2 {
                // ABCxCD -> ABD
                // Reshape
                let w = rhs.permute((1, 0));

                // Broadcasted Multiply
                let mul = self.expand(2, d) * w.expand(0, a).expand(1, b);

                // Sum Reduce
                mul.sum_reduce(3)
            } else if rhs.shape.len() == 3 {
                // Reshape
                let w = rhs.permute((0, 2, 1));

                // Broadcasted Multiply
                let mul = self.expand(2, d) * w.expand(1, b);

                // Sum Reduce
                mul.sum_reduce(3)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.shape(),
                    rhs.shape()
                )
            }
        } else if self.shape.len() == 4 && rhs.shape.len() == 4 {
            // ABCDxABDE -> ABCE
            let a = rhs.shape()[0].small();
            let b = rhs.shape()[1].small();
            let d = rhs.shape()[2].small();
            let e = rhs.shape()[3].small();
            let c = self.shape()[2].small();
            // Reshape
            let s = self.reshape((a * b, c, d));
            let w = rhs.reshape((a * b, d, e)).permute((0, 2, 1));

            // Broadcasted Multiply
            let mul = s.expand(2, e) * w.expand(1, c);

            // Sum Reduce
            mul.sum_reduce(3).reshape((a, b, c, e))
        } else if self.shape.len() == 5 && rhs.shape.len() == 5 {
            // ABCDExABCEF -> ABCDF
            let a = rhs.shape()[0].small();
            let b = rhs.shape()[1].small();
            let c = rhs.shape()[2].small();
            let e = rhs.shape()[3].small();
            let f = rhs.shape()[4].small();
            let d = self.shape()[3].small();
            // Reshape
            let w = rhs.reshape((a * b * c, e, f)).permute((0, 2, 1));
            let s = self.reshape((a * b * c, d, e));

            // Broadcasted Multiply
            let mul = s.expand(2, f) * w.expand(1, d);

            // Sum Reduce
            mul.sum_reduce(3).reshape((a, b, c, d, f))
        } else {
            panic!(
                "Can't matmul lhs {:?} and rhs {:?}",
                self.shape(),
                rhs.shape()
            )
        }
    }

    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor) -> GraphTensor {
        (self * rhs).sum_reduce(0)
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(3), random_vec(6));
        let a = cx.tensor(3).set(a_vec.clone());
        let b = cx.tensor((3, 2)).set(b_vec.clone());
        let mut c = a.matmul(b).retrieve();

        cx.compile(GenericCompiler::default(), &mut c);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_vec, (DConst::<3>,));
        let d_b = d_dev.tensor_from_vec(b_vec, (DConst::<3>, DConst::<2>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let (a_data, b_data) = (random_vec(6), random_vec(9));
        let a = cx.tensor((2, 3));
        a.set(a_data.clone());
        let b = cx.tensor((3, 3));
        b.set(b_data.clone());
        let c = a.matmul(b);
        c.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<3>, DConst::<3>));
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
        let (a_data, b_data) = (random_vec(6), random_vec(6));
        let a = cx.tensor((1, 2, 3));
        a.set(a_data.clone());
        let b = cx.tensor((1, 2, 3));
        b.set(b_data.clone());
        let c = a.matmul(b.permute((0, 2, 1)));
        c.retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<1>, DConst::<2>, DConst::<3>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<1>, DConst::<2>, DConst::<3>));
        let d_c = d_a.matmul(d_b.permute::<Rank3<1, 3, 2>, DAxes3<0, 2, 1>>());

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_batch_matmul2() {
        let mut cx = Graph::new();
        let (a_data, b_data) = (random_vec(4), random_vec(6));
        let a = cx.tensor(('a', 'b'));
        a.set_dyn(a_data.clone(), (2, 2));
        let a = a.expand(0, 1);
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
