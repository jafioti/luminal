use crate::{
    op::{Constant, ConstantValue},
    prelude::{symbolic::BigExpression, *},
};

impl Graph {
    pub fn constant(&mut self, i: f32) -> GraphTensor<R0> {
        GraphTensor::from_id(
            self.add_op(Constant(ConstantValue::Float(i), &self.dyn_map))
                .finish(),
            ShapeTracker::new(&[]),
            self,
        )
    }

    pub fn constant_expr(&mut self, expr: BigExpression) -> GraphTensor<R0> {
        GraphTensor::from_id(
            self.add_op(Constant(
                ConstantValue::Expression(expr.minimize()),
                &self.dyn_map,
            ))
            .finish(),
            ShapeTracker::new(&[]),
            self,
        )
    }

    /// ARange from 0 to N
    pub fn arange<N: Dimension>(&mut self) -> GraphTensor<(N,)> {
        self.constant(1.).expand().cumsum_last_dim() - 1.
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
    // #[test]
    // fn test_dyn_arange() {
    //     let mut cx = Graph::new();

    //     let arange = cx.arange::<Dyn<'a'>>().retrieve();
    //     cx.set_dyn_dim('a', 6);

    //     cx.execute();

    //     assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5.]);
    // }
}
