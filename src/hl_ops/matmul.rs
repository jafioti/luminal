use crate::prelude::*;

pub trait Matmul<S: Shape> {
    type Output;
    fn matmul(self, rhs: GraphTensor<S>) -> Self::Output;
}

// ABxBC -> AC
impl<A: Dimension, B: Dimension, C: Dimension> Matmul<(B, C)> for GraphTensor<(A, B)> {
    type Output = GraphTensor<(A, C)>;
    fn matmul(self, rhs: GraphTensor<(B, C)>) -> Self::Output {
        // Broadcasted Multiply
        let mul = self.expand::<(A, C, B), _>()
            * rhs.permute::<_, Axes2<1, 0>>().expand::<(A, C, B), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<2>>()
    }
}

// AxAB -> B
impl<A: Dimension, B: Dimension> Matmul<(A, B)> for GraphTensor<(A,)> {
    type Output = GraphTensor<(B,)>;
    fn matmul(self, rhs: GraphTensor<(A, B)>) -> Self::Output {
        let s: GraphTensor<(Const<1>, A)> = self.expand();

        // Run normal matmul
        let r = s.matmul(rhs);

        // Sum Reduce
        r.sum_reduce::<_, Axis<0>>()
    }
}

// ABCxCD -> ABD
impl<A: Dimension, B: Dimension, C: Dimension, D: Dimension> Matmul<(C, D)>
    for GraphTensor<(A, B, C)>
{
    type Output = GraphTensor<(A, B, D)>;
    fn matmul(self, rhs: GraphTensor<(C, D)>) -> Self::Output {
        // Reshape
        let w: GraphTensor<(D, C)> = rhs.permute::<_, Axes2<1, 0>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCxACD -> ABD
impl<A: Dimension, B: Dimension, C: Dimension, D: Dimension> Matmul<(A, C, D)>
    for GraphTensor<(A, B, C)>
{
    type Output = GraphTensor<(A, B, D)>;
    fn matmul(self, rhs: GraphTensor<(A, C, D)>) -> Self::Output {
        // Reshape
        let w: GraphTensor<(A, D, C)> = rhs.permute::<_, Axes3<0, 2, 1>>();

        // Broadcasted Multiply
        let mul = self.expand::<(A, B, D, C), _>() * w.expand::<(A, B, D, C), _>();

        // Sum Reduce
        mul.sum_reduce::<_, Axis<3>>()
    }
}

// ABCDxABDE -> ABCE
impl<A: Dimension, B: Dimension, C: Dimension, D: Dimension, E: Dimension> Matmul<(A, B, D, E)>
    for GraphTensor<(A, B, C, D)>
{
    type Output = GraphTensor<(A, B, C, E)>;
    fn matmul(self, rhs: GraphTensor<(A, B, D, E)>) -> Self::Output {
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
    crate::test_imports!();

    #[test]
    fn test_matrix_vector() {
        let mut cx = Graph::new();
        let (a_vec, b_vec) = (random_vec(3), random_vec(6));
        let a = cx.new_tensor::<R1<3>>("Input");
        a.set(a_vec.clone());
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(b_vec.clone());
        let c = a.matmul(b);
        c.mark();

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
        let a = cx.new_tensor::<R2<2, 3>>("Input");
        a.set(a_data.clone());
        let b = cx.new_tensor::<R2<3, 3>>("Input");
        b.set(b_data.clone());
        let c = a.matmul(b);
        c.mark();

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
        let a = cx.new_tensor::<R3<2, 3, 2>>("Input");
        a.set(a_data.clone());
        let b = cx.new_tensor::<R2<2, 4>>("Input");
        b.set(b_data.clone());
        let c = a.matmul(b);
        c.mark();

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
        let a = cx.new_tensor::<R3<1, 2, 3>>("Input");
        a.set(a_data.clone());
        let b = cx.new_tensor::<R3<1, 2, 3>>("Input");
        b.set(b_data.clone());
        let c: GraphTensor<R3<1, 2, 2>> = a.matmul(b.permute::<R3<1, 3, 2>, _>());
        c.mark();

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
        let a = cx.new_tensor::<(Dyn<'a'>, Dyn<'b'>)>("Input");
        a.set_dyn(a_data.clone(), vec![2, 2]);
        let a = a.expand::<(LConst<1>, Dyn<'a'>, Dyn<'b'>), _>();
        let b = cx.new_tensor::<(LConst<1>, Dyn<'b'>, LConst<3>)>("Input");
        b.set_dyn(b_data.clone(), vec![1, 2, 3]);
        let c = a.matmul(b);
        c.mark();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<1>, DConst::<2>, DConst::<2>));
        let d_b = d_dev.tensor_from_vec(b_data, (DConst::<1>, DConst::<2>, DConst::<3>));
        let d_c = d_a.matmul(d_b);

        assert_close(&c.dyn_data(&cx.dyn_map), &d_c.as_vec());
    }
}
