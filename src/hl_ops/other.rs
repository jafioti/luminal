use crate::{
    op::{self, Constant, ConstantValue},
    prelude::{symbolic::BigExpression, *},
};

use self::symbolic::Expression;

impl<S: Shape> GraphTensor<S> {
    /// Cumulative sum last dimension
    pub fn cumsum_last_dim(mut self) -> Self {
        let axis = self.shape.len() - 1;
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        // Pad out length
        let orig_length = self.shape.dims[self.shape.indexes[axis]];
        self.shape.padding[self.shape.indexes[axis]].0 = orig_length - 1;
        self = self.contiguous();

        // Pool
        let mut pooled = self.pool_last_dim::<()>(orig_length, 1.into(), 0);
        // Sum Reduce along new dimension
        let final_id = self
            .graph()
            .add_op(op::SumReduce(axis))
            .input(pooled.id, 0, pooled.shape)
            .finish();
        pooled.shape.remove_dim(axis + 1);
        GraphTensor::from_id(final_id, pooled.shape, self.graph_ref)
    }

    /// Cumulative product last dimension
    pub fn cumprod_last_dim(self) -> Self {
        self.ln().cumsum_last_dim().exp()
    }
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        ConstantValue::Float(value)
    }
}
impl From<f64> for ConstantValue {
    fn from(value: f64) -> Self {
        ConstantValue::Float(value as f32)
    }
}
impl From<BigExpression> for ConstantValue {
    fn from(value: BigExpression) -> Self {
        ConstantValue::Expression(value)
    }
}
impl From<Expression> for ConstantValue {
    fn from(value: Expression) -> Self {
        ConstantValue::Expression(value.into())
    }
}
impl From<&BigExpression> for ConstantValue {
    fn from(value: &BigExpression) -> Self {
        ConstantValue::Expression(value.clone())
    }
}
impl From<&Expression> for ConstantValue {
    fn from(value: &Expression) -> Self {
        ConstantValue::Expression((*value).into())
    }
}

impl Graph {
    /// A scalar constant
    pub fn constant(&mut self, i: impl Into<ConstantValue>) -> GraphTensor<R0> {
        GraphTensor::from_id(
            self.add_op(Constant(i.into(), &self.dyn_map)).finish(),
            ShapeTracker::new(&[]),
            self,
        )
    }

    /// A scalar constant evaluated from an expression at runtime
    pub fn constant_expr<E: Into<BigExpression>>(&mut self, expr: E) -> GraphTensor<R0> {
        GraphTensor::from_id(
            self.add_op(Constant(
                ConstantValue::Expression(expr.into().minimize()),
                &self.dyn_map,
            ))
            .finish(),
            ShapeTracker::new(&[]),
            self,
        )
    }

    /// ARange from 0 to N
    pub fn arange<N: Dimension>(&mut self) -> GraphTensor<(N,)> {
        if N::const_size()
            .to_usize()
            .map(|i| i == 1)
            .unwrap_or_default()
        {
            // Single number ARange is just 0
            self.constant(0.).expand()
        } else {
            self.constant(1.).expand().cumsum_last_dim() - 1.
        }
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril<S: Dimension>(&mut self, diagonal: i32) -> GraphTensor<(S, S)> {
        let horizontal = self.arange::<S>().expand::<(S, S), Axis<0>>();
        let vertical = self.arange::<S>().expand::<(S, S), Axis<1>>();

        (horizontal
            + self
                .constant(-(diagonal as f32 + 1.))
                .expand_to(horizontal.shape))
        .less_than(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu<S: Dimension>(&mut self, diagonal: i32) -> GraphTensor<(S, S)> {
        let horizontal = self.arange::<S>().expand::<(S, S), Axis<0>>();
        let vertical = self.arange::<S>().expand::<(S, S), Axis<1>>();

        (horizontal
            + self
                .constant(-(diagonal as f32 - 1.))
                .expand_to(horizontal.shape))
        .greater_than(vertical)
    }
}

impl<S: Dimension, const DIM: usize> GraphTensor<(S, Const<DIM>)> {
    /// Gather a batch of vectors from a matrix
    pub fn gather<B: Dimension>(self, indexes: GraphTensor<(B,)>) -> GraphTensor<(B, Const<DIM>)> {
        let one_hot = indexes
            .graph()
            .arange::<S>()
            .expand::<(B, S), _>()
            .equals(indexes.expand());
        (one_hot.expand::<(B, S, Const<DIM>), _>() * self.expand()).sum_reduce::<_, Axis<1>>()
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();
    #[test]
    fn test_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange::<LConst<10>>().retrieve();
        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }

    #[test]
    fn test_cumprod() {
        let mut cx = Graph::new();

        let a = cx.tensor::<R1<3>>().set(vec![3., 2., 5.]);
        let b = a.cumprod_last_dim().retrieve();
        cx.execute();

        assert_close(&b.data(), &[3., 6., 30.]);
    }

    #[test]
    fn test_dyn_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange::<Dyn<'a'>>().retrieve();
        cx.set_dyn_dim('a', 6);

        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_tril() {
        let mut cx = Graph::new();

        let triangle = cx.tril::<LConst<5>>(1).retrieve();

        cx.execute();

        assert_exact(
            &triangle.data(),
            &[
                [1.00, 1.00, 0.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_triu() {
        let mut cx = Graph::new();

        let a = cx.triu::<LConst<3>>(-1).retrieve();
        let b = cx.triu::<LConst<3>>(0).retrieve();
        let c = cx.triu::<LConst<3>>(1).retrieve();

        cx.execute();

        assert_exact(
            &a.data(),
            &[[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [0.00, 1.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &b.data(),
            &[[1.00, 1.00, 1.00], [0.00, 1.00, 1.00], [0.00, 0.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &c.data(),
            &[[0.00, 1.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 0.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
    }
}
