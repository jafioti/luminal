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
}
