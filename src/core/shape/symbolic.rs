// Super minimal symbolic algebra library

use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Sub},
};

use tinyvec::ArrayVec;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Expression<const S: usize = 20>
// We need to figure out how to reduce this, can't be fixed at 20. ShapeTracker would take up 6 dims * 12 pads * 12 slices * 20 terms * 8 bytes = 138kb
where
    tinyvec::ArrayVec<[Term; S]>: Copy,
    [Term; S]: tinyvec::Array,
    <[Term; S] as tinyvec::Array>::Item: std::cmp::Eq,
    <[Term; S] as tinyvec::Array>::Item: std::hash::Hash,
{
    pub terms: ArrayVec<[Term; S]>,
}

impl Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for term in &self.terms {
            term.fmt(f)?;
            write!(f, " ")?;
        }
        Ok(())
    }
}

impl Expression {
    pub fn exec(&self, vars: &HashMap<char, usize>) -> Option<usize> {
        let mut stack = ArrayVec::<[usize; 10]>::new();
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n),
                Term::Var(c) => {
                    if let Some(n) = vars.get(c) {
                        stack.push(*n)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop()
    }

    pub fn minimize(mut self) -> Self {
        let mut i = 0;
        while i < self.terms.len().saturating_sub(2) {
            match (self.terms[i], self.terms[i + 1], self.terms[i + 2].as_op()) {
                (Term::Num(b), Term::Num(a), Some(term)) => {
                    self.terms[i] = Term::Num(term(a, b));
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                _ => {
                    i += 1;
                }
            }
        }

        self
    }

    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&HashMap::default())
    }
    pub fn to_symbol(&self) -> Option<char> {
        match self.terms[0] {
            Term::Var(c) => Some(c),
            _ => None,
        }
    }

    pub fn min<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.minimize()
    }

    pub fn max<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.minimize()
    }

    pub fn gte<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.minimize()
    }

    pub fn lt<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs.minimize()
    }

    pub fn is_unknown(&self) -> bool {
        self.terms.iter().any(|t| matches!(t, Term::Var('-')))
    }
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct BigExpression {
    pub terms: Vec<Term>,
}

impl Debug for BigExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for term in &self.terms {
            term.fmt(f)?;
            write!(f, " ")?;
        }
        Ok(())
    }
}

impl BigExpression {
    pub fn exec(&self, vars: &HashMap<char, usize>) -> Option<usize> {
        let mut stack = Vec::new();
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n),
                Term::Var(c) => {
                    if let Some(n) = vars.get(c) {
                        stack.push(*n)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop()
    }

    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&HashMap::default())
    }
    pub fn to_symbol(&self) -> Option<char> {
        match self.terms[0] {
            Term::Var(c) => Some(c),
            _ => None,
        }
    }

    pub fn min<E: Into<BigExpression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs
    }

    pub fn max<E: Into<BigExpression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs
    }

    pub fn gte<E: Into<BigExpression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs
    }

    pub fn lt<E: Into<BigExpression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Term {
    Num(usize),
    Var(char),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
}

impl Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Num(n) => write!(f, "{n}"),
            Term::Var(c) => write!(f, "{c}"),
            Term::Add => write!(f, "+"),
            Term::Sub => write!(f, "-"),
            Term::Mul => write!(f, "*"),
            Term::Div => write!(f, "/"),
            Term::Mod => write!(f, "%"),
            Term::Min => write!(f, "min"),
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self::Num(0)
    }
}

impl Term {
    pub fn big_expr(self) -> BigExpression {
        self.into()
    }

    pub fn expr(self) -> Expression {
        self.into()
    }

    pub fn as_op(self) -> Option<fn(usize, usize) -> usize> {
        match self {
            Term::Add => Some(std::ops::Add::add),
            Term::Sub => Some(std::ops::Sub::sub),
            Term::Mul => Some(std::ops::Mul::mul),
            Term::Div => Some(std::ops::Div::div),
            Term::Mod => Some(std::ops::Rem::rem),
            Term::Max => Some(core::cmp::Ord::max),
            Term::Min => Some(core::cmp::Ord::min),
            Term::And => Some(|a, b| (a != 0 && b != 0) as usize),
            Term::Or => Some(|a, b| (a != 0 || b != 0) as usize),
            Term::Gte => Some(|a, b| (a >= b) as usize),
            Term::Lt => Some(|a, b| (a < b) as usize),
            _ => None,
        }
    }
}

impl From<Term> for BigExpression {
    fn from(value: Term) -> Self {
        BigExpression { terms: vec![value] }
    }
}

impl From<Term> for Expression {
    fn from(value: Term) -> Self {
        let mut terms = ArrayVec::new();
        terms.push(value);
        Expression { terms }
    }
}

pub trait ExprInterface {
    fn expr(self) -> Expression;
}

impl ExprInterface for usize {
    fn expr(self) -> Expression {
        Term::Num(self).expr()
    }
}

impl ExprInterface for &usize {
    fn expr(self) -> Expression {
        Term::Num(*self).expr()
    }
}

impl ExprInterface for char {
    fn expr(self) -> Expression {
        Term::Var(self).expr()
    }
}

impl ExprInterface for i32 {
    fn expr(self) -> Expression {
        Term::Num(self as usize).expr()
    }
}

impl ExprInterface for Expression {
    fn expr(self) -> Expression {
        self
    }
}

pub trait BigExprInterface {
    fn big_expr(self) -> BigExpression;
}

impl BigExprInterface for usize {
    fn big_expr(self) -> BigExpression {
        Term::Num(self).big_expr()
    }
}

impl BigExprInterface for char {
    fn big_expr(self) -> BigExpression {
        Term::Var(self).big_expr()
    }
}

impl BigExprInterface for i32 {
    fn big_expr(self) -> BigExpression {
        Term::Num(self as usize).big_expr()
    }
}

impl BigExprInterface for Expression {
    fn big_expr(self) -> BigExpression {
        self.into()
    }
}

impl BigExprInterface for BigExpression {
    fn big_expr(self) -> BigExpression {
        self
    }
}

impl From<Expression> for BigExpression {
    fn from(value: Expression) -> Self {
        Self {
            terms: value.terms.to_vec(),
        }
    }
}

impl<E: Into<Expression>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<E: Into<Expression>> BitOr<E> for Expression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.minimize()
    }
}

impl<E: Into<BigExpression>> Add<E> for BigExpression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs
    }
}

impl<E: Into<BigExpression>> Sub<E> for BigExpression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs
    }
}

impl<E: Into<BigExpression>> Mul<E> for BigExpression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs
    }
}

impl<E: Into<BigExpression>> Div<E> for BigExpression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs
    }
}

impl<E: Into<BigExpression>> Rem<E> for BigExpression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs
    }
}

impl<E: Into<BigExpression>> BitAnd<E> for BigExpression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs
    }
}

impl<E: Into<BigExpression>> BitOr<E> for BigExpression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = ('x'.expr() + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);

        let n = ('x'.big_expr() + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);
    }
}
