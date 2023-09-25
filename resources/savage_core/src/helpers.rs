// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

//! Operators, conversions, and helper functions to make working with expressions easier.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

use num::{One, Zero};

use crate::expression::{
    Complex, Expression, Integer, Matrix, Rational, RationalRepresentation, Type, Vector,
};

impl Expression {
    pub fn min(self, other: Self) -> Self {
        Expression::Min(Box::new(self), Box::new(other))
    }
}

impl Neg for Expression {
    type Output = Self;

    fn neg(self) -> Self {
        Expression::Negation(Box::new(self))
    }
}

impl Not for Expression {
    type Output = Self;

    fn not(self) -> Self {
        Expression::Not(Box::new(self))
    }
}

impl Add for Expression {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Expression::Sum(Box::new(self), Box::new(other))
    }
}

impl AddAssign for Expression {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl Sub for Expression {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Expression::Difference(Box::new(self), Box::new(other))
    }
}

impl SubAssign for Expression {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl Mul for Expression {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Expression::Product(Box::new(self), Box::new(other))
    }
}

impl MulAssign for Expression {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl Div for Expression {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Expression::Quotient(Box::new(self), Box::new(other))
    }
}

impl DivAssign for Expression {
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

impl Rem for Expression {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Expression::Remainder(Box::new(self), Box::new(other))
    }
}

impl RemAssign for Expression {
    fn rem_assign(&mut self, other: Self) {
        *self = self.clone() % other;
    }
}

impl From<&Self> for Expression {
    fn from(expression: &Self) -> Self {
        expression.clone()
    }
}

impl From<Integer> for Expression {
    fn from(integer: Integer) -> Self {
        Expression::Integer(integer)
    }
}

impl TryFrom<Expression> for Integer {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Type::Number(z, _) = expression.typ() {
            if z.im.is_zero() && z.re.denom().is_one() {
                Ok(z.re.numer().clone())
            } else {
                Err(expression)
            }
        } else {
            Err(expression)
        }
    }
}

impl From<Rational> for Expression {
    fn from(rational: Rational) -> Self {
        Expression::Rational(rational, RationalRepresentation::Fraction)
    }
}

impl TryFrom<Expression> for Rational {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Type::Number(z, _) = expression.typ() {
            if z.im.is_zero() {
                Ok(z.re)
            } else {
                Err(expression)
            }
        } else {
            Err(expression)
        }
    }
}

impl From<Complex> for Expression {
    fn from(complex: Complex) -> Self {
        Expression::Complex(complex, RationalRepresentation::Fraction)
    }
}

impl TryFrom<Expression> for Complex {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Type::Number(z, _) = expression.typ() {
            Ok(z)
        } else {
            Err(expression)
        }
    }
}

impl From<Vector> for Expression {
    fn from(vector: Vector) -> Self {
        Expression::Vector(vector)
    }
}

impl TryFrom<Expression> for Vector {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Type::Matrix(m) = expression.typ() {
            if m.ncols() == 1 {
                Ok(m.column(0).clone_owned())
            } else {
                Err(expression)
            }
        } else {
            Err(expression)
        }
    }
}

impl From<Matrix> for Expression {
    fn from(matrix: Matrix) -> Self {
        Expression::Matrix(matrix)
    }
}

impl TryFrom<Expression> for Matrix {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Type::Matrix(m) = expression.typ() {
            Ok(m)
        } else {
            Err(expression)
        }
    }
}

impl From<bool> for Expression {
    fn from(boolean: bool) -> Self {
        Expression::Boolean(boolean)
    }
}

impl TryFrom<Expression> for bool {
    type Error = Expression;

    fn try_from(expression: Expression) -> Result<Self, Self::Error> {
        if let Expression::Boolean(boolean) = expression {
            Ok(boolean)
        } else {
            Err(expression)
        }
    }
}

/// Returns an expression representing the variable with the given identifier.
pub fn var(identifier: impl Into<String>) -> Expression {
    Expression::Variable(identifier.into())
}

/// Returns an expression representing the value of the given function at the given arguments.
pub fn fun(function: impl Into<Expression>, arguments: impl Into<Vec<Expression>>) -> Expression {
    Expression::FunctionValue(Box::new(function.into()), arguments.into())
}

/// Returns an expression representing the given integer.
pub fn int(integer: impl Into<Integer>) -> Expression {
    Expression::Integer(integer.into())
}

/// Returns an expression representing the rational number with
/// the given numerator and denominator, using fraction representation.
pub fn rat(numerator: impl Into<Integer>, denominator: impl Into<Integer>) -> Expression {
    Expression::Rational(
        Rational::new(numerator.into(), denominator.into()),
        RationalRepresentation::Fraction,
    )
}

/// Returns an expression representing the rational number with
/// the given numerator and denominator, using decimal representation,
/// falling back to fraction representation if the number cannot be
/// represented as a finite decimal.
pub fn ratd(numerator: impl Into<Integer>, denominator: impl Into<Integer>) -> Expression {
    Expression::Rational(
        Rational::new(numerator.into(), denominator.into()),
        RationalRepresentation::Decimal,
    )
}

/// Returns an expression representing the complex number with
/// real and imaginary parts being rational numbers described by
/// the given numerators and denominators, using fraction representation.
pub fn com(
    real_numerator: impl Into<Integer>,
    real_denominator: impl Into<Integer>,
    imaginary_numerator: impl Into<Integer>,
    imaginary_denominator: impl Into<Integer>,
) -> Expression {
    Expression::Complex(
        Complex::new(
            Rational::new(real_numerator.into(), real_denominator.into()),
            Rational::new(imaginary_numerator.into(), imaginary_denominator.into()),
        ),
        RationalRepresentation::Fraction,
    )
}

/// Returns an expression representing the complex number with
/// real and imaginary parts being rational numbers described by
/// the given numerators and denominators, using decimal representation,
/// falling back to fraction representation for parts that cannot be
/// represented as a finite decimal.
pub fn comd(
    real_numerator: impl Into<Integer>,
    real_denominator: impl Into<Integer>,
    imaginary_numerator: impl Into<Integer>,
    imaginary_denominator: impl Into<Integer>,
) -> Expression {
    Expression::Complex(
        Complex::new(
            Rational::new(real_numerator.into(), real_denominator.into()),
            Rational::new(imaginary_numerator.into(), imaginary_denominator.into()),
        ),
        RationalRepresentation::Decimal,
    )
}

/// Returns an expression representing the first expression raised to the power of the second.
pub fn pow(base: impl Into<Expression>, exponent: impl Into<Expression>) -> Expression {
    Expression::Power(Box::new(base.into()), Box::new(exponent.into()))
}

/// Returns an expression representing whether two expressions are equal.
pub fn eq(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::Equal(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing whether two expressions are not equal.
pub fn ne(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::NotEqual(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing whether the first expression is less than the second.
pub fn lt(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::LessThan(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing whether the first expression is less than or equal to the second.
pub fn le(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::LessThanOrEqual(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing whether the first expression is greater than the second.
pub fn gt(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::GreaterThan(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing whether the first expression is greater than or equal to the second.
pub fn ge(left: impl Into<Expression>, right: impl Into<Expression>) -> Expression {
    Expression::GreaterThanOrEqual(Box::new(left.into()), Box::new(right.into()))
}

/// Returns an expression representing the logical conjunction (AND) of two expressions.
pub fn and(a: impl Into<Expression>, b: impl Into<Expression>) -> Expression {
    Expression::And(Box::new(a.into()), Box::new(b.into()))
}

/// Returns an expression representing the logical disjunction (OR) of two expressions.
pub fn or(a: impl Into<Expression>, b: impl Into<Expression>) -> Expression {
    Expression::Or(Box::new(a.into()), Box::new(b.into()))
}
