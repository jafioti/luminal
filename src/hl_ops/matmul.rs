use crate::prelude::*;

// ABxBC -> AC
impl<A: Dimension, B: Dimension> GraphTensor<(A, B)> {
    pub fn matmul<C: Dimension>(self, rhs: GraphTensor<(B, C)>) -> GraphTensor<(A, C)> {
        // Reshape
        let w: GraphTensor<(C, B)> = rhs.permute::<_, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, C, B), _>() * w.expand::<(A, C, B), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<2>>()
    }
}

// AxAB -> B
impl<A: Dimension> GraphTensor<(A,)> {
    pub fn matmul<B: Dimension>(self, rhs: GraphTensor<(A, B)>) -> GraphTensor<(B,)> {
        let s: GraphTensor<(Const<1>, A)> = self.expand();

        // Run normal matmul
        let r = s.matmul(rhs);

        // Sum Reduce
        r.sum_reduce::<_, Axis<0>>()
    }
}

// ABCxCD -> ABD
impl<A: Dimension, B: Dimension, C: Dimension> GraphTensor<(A, B, C)> {
    pub fn matmul<D: Dimension>(self, rhs: GraphTensor<(C, D)>) -> GraphTensor<(A, B, D)> {
        // Reshape
        let w: GraphTensor<(D, C)> = rhs.permute::<_, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCxACD -> ABD
impl<A: Dimension, B: Dimension, C: Dimension> GraphTensor<(A, B, C)> {
    pub fn batch_matmul<D: Dimension>(self, rhs: GraphTensor<(A, C, D)>) -> GraphTensor<(A, B, D)> {
        // Reshape
        let w: GraphTensor<(A, D, C)> = rhs.permute::<_, Axes3<0, 2, 1>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCDxABDE -> ABCE
impl<A: Dimension, B: Dimension, C: Dimension, D: Dimension> GraphTensor<(A, B, C, D)> {
    pub fn batch_matmul<E: Dimension>(
        self,
        rhs: GraphTensor<(A, B, D, E)>,
    ) -> GraphTensor<(A, B, C, E)> {
        // Reshape
        let w: GraphTensor<(A, B, E, D)> = rhs.permute::<_, Axes4<0, 1, 3, 2>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, C, E, D), _>() * w.expand::<(A, B, C, E, D), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<4>>()
    }
}

impl<A: Dimension> GraphTensor<(A,)> {
    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor<(A,)>) -> GraphTensor<R0> {
        (self * rhs).sum_reduce()
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close_data};
    use dfdx::prelude::*;

    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(vec![1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(vec![1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1., 2., 3.]);
        let d_b = d_dev.tensor([[1., 2.], [3., 1.], [2., 3.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<3, 3>>("Input");
        b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([[1., 2., 3.], [1., 2., 3.]]);
        let d_b = d_dev.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<2, 3, 2>>("Input");
        a.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = cx.new_tensor::<R2<2, 4>>("Input");
        b.set(vec![1., 2., 3., 1., 1., 2., 3., 1.]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([
            [[1., 2.], [3., 1.], [2., 3.]],
            [[1., 2.], [3., 1.], [2., 3.]],
        ]);
        let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 3., 1.]]);
        let d_c = d_a.matmul(d_b);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_batch_matmul() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R3<1, 2, 3>>("Input");
        a.set(vec![8.6, 8.0, 12.0, 9.9, 10.0, 15.0]);
        let b = cx.new_tensor::<R3<1, 2, 3>>("Input");
        b.set(vec![4.0, -12.0, 12.0, 5.0, 70.0, 15.0]);
        let c = a.batch_matmul(b.permute());
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![8.6, 8.0, 12.0, 9.9, 10.0, 15.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![4.0, -12.0, 12.0, 5.0, 70.0, 15.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_c = d_a.matmul(d_b.permute::<Rank3<1, 3, 2>, dfdx::shapes::Axes3<0, 2, 1>>());

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_batch_batch_matmul2() {
        let mut cx = Graph::new();
        let mut a = cx.new_tensor::<(usize, usize)>("Input");
        a.set_dyn(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
        let a = a.expand::<(crate::shape::Const<1>, usize, usize), _>();
        let mut b =
            cx.new_tensor::<(crate::shape::Const<1>, usize, crate::shape::Const<3>)>("Input");
        b.set_dyn(vec![32.0, -2.0, 0.0, -17.0, 40.0, -3.0], vec![1, 2, 3]);
        let c = a.batch_matmul(b);
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![0.0, 1.0, 0.0, 1.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<2>,
            ),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![32.0, -2.0, 0.0, -17.0, 40.0, -3.0],
            (
                dfdx::shapes::Const::<1>,
                dfdx::shapes::Const::<2>,
                dfdx::shapes::Const::<3>,
            ),
        );
        let d_c = d_a.matmul(d_b);

        assert_close_data(&c.data(), &d_c.as_vec());
    }
}
