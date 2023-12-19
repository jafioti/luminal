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

    /// Lower left-hand triangle of 1s
    pub fn tril<H: Dimension, W: Dimension>(&mut self, offset: i32) -> GraphTensor<(H, W)> {
        let horizontal = self.arange::<W>().expand::<(H, W), _>();
        let vertical = self.arange::<H>().expand::<(H, W), _>();

        (horizontal + self.constant(-(offset as f32 + 1.)).expand()).less_than(vertical)
    }

    /// Lower left-hand triangle of 1s
    pub fn triu<H: Dimension, W: Dimension>(&mut self, offset: i32) -> GraphTensor<(H, W)> {
        let horizontal = self.arange::<W>().expand::<(H, W), _>();
        let vertical = self.arange::<H>().expand::<(H, W), _>();

        (horizontal + self.constant(-(offset as f32 - 1.)).expand()).greater_than(vertical)
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

        let triangle = cx.tril::<LConst<5>, LConst<5>>(1).retrieve();

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

        let a = cx.triu::<LConst<3>, LConst<3>>(-1).retrieve();
        let b = cx.triu::<LConst<3>, LConst<3>>(0).retrieve();
        let c = cx.triu::<LConst<3>, LConst<3>>(1).retrieve();

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
