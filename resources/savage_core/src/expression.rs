// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2021-2022  Philipp Emanuel Weidmann <pew@worldwidemann.com>

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use derivative::*;
use num::{Signed, Zero};

use crate::evaluate::Error;

/// Function implementation.
pub type Function =
    dyn Fn(&Expression, &[Expression], &HashMap<String, Expression>) -> Result<Expression, Error>;

/// Arbitrary-precision integer.
pub type Integer = num::bigint::BigInt;

/// Arbitrary-precision rational number.
pub type Rational = num::rational::Ratio<Integer>;

/// Arbitrary-precision complex number (i.e. real and imaginary parts are arbitrary-precision rational numbers).
pub type Complex = num::complex::Complex<Rational>;

/// Column vector with expressions as components.
pub type Vector = nalgebra::DVector<Expression>;

/// Column-major matrix with expressions as components.
pub type Matrix = nalgebra::DMatrix<Expression>;

/// Preferred representation when printing a rational number.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum RationalRepresentation {
    /// Fraction (numerator/denominator).
    Fraction,
    /// Decimal, falling back to fraction representation
    /// if the number cannot be represented as a finite decimal.
    Decimal,
}

impl RationalRepresentation {
    /// Returns the preferred representation for the result of an operation
    /// on two numbers with representations `self` and `other`.
    pub(crate) fn merge(self, other: Self) -> Self {
        use RationalRepresentation::*;

        if self == Decimal || other == Decimal {
            Decimal
        } else {
            Fraction
        }
    }
}

/// Symbolic expression.
#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub enum Expression {
    /// Variable with identifier.
    Variable(String),
    /// Function with identifier and implementation.
    Function(
        String,
        #[derivative(PartialEq = "ignore", Debug = "ignore")] Rc<Function>,
    ),
    /// Value of a function expression at the given arguments.
    FunctionValue(Box<Self>, Vec<Self>),
    /// Integer.
    Integer(Integer),
    /// Rational number with preferred representation.
    Rational(Rational, RationalRepresentation),
    /// Complex number with preferred representation for real and imaginary parts.
    Complex(Complex, RationalRepresentation),
    /// Column vector.
    Vector(Vector),
    /// Element of a column vector expression given by an index expression.
    VectorElement(Box<Self>, Box<Self>),
    /// Column-major matrix.
    Matrix(Matrix),
    /// Element of a column-major matrix expression given by row and column index expressions.
    MatrixElement(Box<Self>, Box<Self>, Box<Self>),
    /// Boolean value.
    Boolean(bool),
    /// Arithmetic negation of an expression.
    Negation(Box<Self>),
    /// Logical negation (NOT) of an expression.
    Not(Box<Self>),
    /// Sum of two expressions.
    Sum(Box<Self>, Box<Self>),
    /// Difference of two expressions.
    Difference(Box<Self>, Box<Self>),
    /// Product of two expressions.
    Product(Box<Self>, Box<Self>),
    /// Quotient of two expressions.
    Quotient(Box<Self>, Box<Self>),
    /// Remainder of the Euclidean division of the first expression by the second.
    Remainder(Box<Self>, Box<Self>),
    /// The first expression raised to the power of the second.
    Power(Box<Self>, Box<Self>),
    /// Whether two expressions are equal.
    Equal(Box<Self>, Box<Self>),
    /// Whether two expressions are not equal.
    NotEqual(Box<Self>, Box<Self>),
    /// Whether the first expression is less than the second.
    LessThan(Box<Self>, Box<Self>),
    /// Whether the first expression is less than or equal to the second.
    LessThanOrEqual(Box<Self>, Box<Self>),
    /// Whether the first expression is greater than the second.
    GreaterThan(Box<Self>, Box<Self>),
    /// Whether the first expression is greater than or equal to the second.
    GreaterThanOrEqual(Box<Self>, Box<Self>),
    /// Logical conjunction (AND) of two expressions.
    And(Box<Self>, Box<Self>),
    /// Logical disjunction (OR) of two expressions.
    Or(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
}

/// Basic expression type designed to make evaluating expressions easier.
#[derive(Derivative)]
#[derivative(PartialEq, Eq, Clone, Debug)]
pub(crate) enum Type {
    /// Function with identifier and implementation.
    Function(
        String,
        #[derivative(PartialEq = "ignore", Debug = "ignore")] Rc<Function>,
    ),
    /// Number with preferred representation for rational parts.
    Number(Complex, RationalRepresentation),
    /// Column-major matrix.
    Matrix(Matrix),
    /// Boolean expression with value (if available).
    Boolean(Option<bool>),
    /// Arithmetic expression (in particular, this expression does *not* have a boolean value).
    Arithmetic,
    /// Expression that cannot be assigned to any of the above types with certainty.
    Unknown,
}

/// Associativity of an operator expression.
#[allow(clippy::enum_variant_names)]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(crate) enum Associativity {
    /// `a OP b OP c == (a OP b) OP c`.
    LeftAssociative,
    /// `a OP b OP c == a OP (b OP c)`.
    RightAssociative,
    /// `a OP b OP c == (a OP b) OP c == a OP (b OP c)`.
    Associative,
}

impl Expression {
    /// Returns the basic type of the expression.
    pub(crate) fn typ(&self) -> Type {
        use Expression::*;
        use RationalRepresentation::*;
        use Type::{
            Arithmetic, Boolean as Bool, Function as Fun, Matrix as Mat, Number as Num, Unknown,
        };

        match self {
            Variable(_) => Unknown,
            Function(identifier, f) => Fun(identifier.clone(), f.clone()),
            FunctionValue(_, _) => Unknown,
            Integer(n) => Num(self::Rational::from_integer(n.clone()).into(), Fraction),
            Rational(x, representation) => Num(x.into(), *representation),
            Complex(z, representation) => Num(z.clone(), *representation),
            Vector(v) => Mat(self::Matrix::from_columns(&[v.clone()])),
            VectorElement(_, _) => Unknown,
            Matrix(m) => Mat(m.clone()),
            MatrixElement(_, _, _) => Unknown,
            Boolean(boolean) => Bool(Some(*boolean)),
            Negation(_) => Arithmetic,
            Not(_) => Bool(None),
            Sum(_, _) => Arithmetic,
            Difference(_, _) => Arithmetic,
            Product(_, _) => Arithmetic,
            Quotient(_, _) => Arithmetic,
            Remainder(_, _) => Arithmetic,
            Power(_, _) => Arithmetic,
            Equal(_, _) => Bool(None),
            NotEqual(_, _) => Bool(None),
            LessThan(_, _) => Bool(None),
            LessThanOrEqual(_, _) => Bool(None),
            GreaterThan(_, _) => Bool(None),
            GreaterThanOrEqual(_, _) => Bool(None),
            And(_, _) => Bool(None),
            Or(_, _) => Bool(None),
            Min(_, _) => Arithmetic,
            Max(_, _) => Arithmetic,
        }
    }

    /// Returns the precedence (as an integer intended for comparison)
    /// and associativity of the expression. For unary or non-operator
    /// expressions, to which the concept of associativity doesn't apply,
    /// `Associative` is returned.
    pub(crate) fn precedence_and_associativity(&self) -> (isize, Associativity) {
        use Associativity::*;
        use Expression::*;

        match self {
            Variable(_) => (isize::MAX, Associative),
            Function(_, _) => (isize::MAX, Associative),
            FunctionValue(_, _) => (5, Associative),
            Integer(n) => {
                if n.is_negative() {
                    (2, Associative)
                } else {
                    (isize::MAX, Associative)
                }
            }
            Rational(x, _) => {
                if self.to_string().contains('/') {
                    (2, LeftAssociative)
                } else if x.is_negative() {
                    (2, Associative)
                } else {
                    (isize::MAX, Associative)
                }
            }
            Complex(z, _) => {
                if !z.re.is_zero() && !z.im.is_zero() {
                    if self.to_string().contains('+') {
                        (1, Associative)
                    } else {
                        (1, LeftAssociative)
                    }
                } else if self.to_string().contains('/') {
                    (2, LeftAssociative)
                } else if z.re.is_negative() || !z.im.is_zero() {
                    (2, Associative)
                } else {
                    (isize::MAX, Associative)
                }
            }
            Vector(_) => (isize::MAX, Associative),
            VectorElement(_, _) => (5, Associative),
            Matrix(_) => (isize::MAX, Associative),
            MatrixElement(_, _, _) => (5, Associative),
            Boolean(_) => (isize::MAX, Associative),
            Negation(_) => (3, Associative),
            Not(_) => (3, Associative),
            Sum(_, _) => (1, Associative),
            Min(_, _) => (0, Associative),
            Max(_, _) => (0, Associative),
            Difference(_, _) => (1, LeftAssociative),
            Product(_, _) => (2, Associative),
            Quotient(_, _) => (2, LeftAssociative),
            Remainder(_, _) => (2, LeftAssociative),
            Power(_, _) => (4, RightAssociative),
            Equal(_, _) => (0, Associative),
            NotEqual(_, _) => (0, Associative),
            LessThan(_, _) => (0, Associative),
            LessThanOrEqual(_, _) => (0, Associative),
            GreaterThan(_, _) => (0, Associative),
            GreaterThanOrEqual(_, _) => (0, Associative),
            And(_, _) => (-1, Associative),
            Or(_, _) => (-2, Associative),
        }
    }

    /// Returns the precedence (as an integer intended for comparison) of the expression.
    pub(crate) fn precedence(&self) -> isize {
        self.precedence_and_associativity().0
    }

    /// Returns the associativity of the expression.
    /// For unary or non-operator expressions, to which the concept
    /// of associativity doesn't apply, `Associative` is returned.
    pub(crate) fn associativity(&self) -> Associativity {
        self.precedence_and_associativity().1
    }

    /// Returns all sub-expressions that the expression contains.
    /// The returned list is built recursively and thus includes sub-expressions
    /// of sub-expressions and so on, as well as the expression itself.
    pub(crate) fn parts(&self) -> Vec<Self> {
        use Expression::*;

        let mut parts = vec![self.clone()];

        match self {
            Variable(_) => {}
            Function(_, _) => {}
            FunctionValue(function, arguments) => {
                parts.append(&mut function.parts());

                for argument in arguments {
                    parts.append(&mut argument.parts());
                }
            }
            Integer(_) => {}
            Rational(_, _) => {}
            Complex(_, _) => {}
            Vector(v) => {
                for element in v.iter() {
                    parts.append(&mut element.parts());
                }
            }
            VectorElement(vector, i) => {
                parts.append(&mut vector.parts());
                parts.append(&mut i.parts());
            }
            Matrix(m) => {
                for element in m.iter() {
                    parts.append(&mut element.parts());
                }
            }
            MatrixElement(matrix, i, j) => {
                parts.append(&mut matrix.parts());
                parts.append(&mut i.parts());
                parts.append(&mut j.parts());
            }
            Boolean(_) => {}
            Negation(a) | Not(a) => {
                parts.append(&mut a.parts());
            }
            Sum(a, b)
            | Difference(a, b)
            | Product(a, b)
            | Quotient(a, b)
            | Remainder(a, b)
            | Power(a, b)
            | Equal(a, b)
            | NotEqual(a, b)
            | LessThan(a, b)
            | LessThanOrEqual(a, b)
            | GreaterThan(a, b)
            | GreaterThanOrEqual(a, b)
            | And(a, b)
            | Or(a, b)
            | Min(a, b)
            | Max(a, b) => {
                parts.append(&mut a.parts());
                parts.append(&mut b.parts());
            }
        }

        parts
    }

    /// Returns the identifiers of all variables that the expression contains.
    pub fn variables(&self) -> HashSet<String> {
        let mut identifiers = HashSet::new();

        for part in self.parts() {
            if let Self::Variable(identifier) = part {
                identifiers.insert(identifier);
            }
        }

        identifiers
    }
}
