// Super minimal symbolic algebra library

use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Sub},
};

// use tinyvec::ArrayVec;

#[derive(Clone, Default)]
pub struct Expression {
    pub terms: Vec<Term>,
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
    pub fn exec(&self, vars: &HashMap<char, usize>) -> usize {
        let mut stack = Vec::new();
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n),
                Term::Var(c) => stack.push(vars[c]),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(match term {
                        Term::Add => a + b,
                        Term::Sub => a - b,
                        Term::Mul => a * b,
                        Term::Div => a / b,
                        Term::Mod => a % b,
                        Term::Max => a.max(b),
                        Term::Min => a.min(b),
                        Term::And => (a != 0 && b != 0) as usize,
                        Term::Or => (a != 0 || b != 0) as usize,
                        Term::Gte => (a >= b) as usize,
                        Term::Lt => (a < b) as usize,
                        _ => unreachable!(),
                    });
                }
            }
        }
        stack.pop().unwrap_or_default()
    }

    pub fn min<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs
    }

    pub fn max<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs
    }

    pub fn gte<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs
    }

    pub fn lt<E: Into<Expression>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs
    }
}

#[derive(Clone, Copy)]
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
    pub fn expr(self) -> Expression {
        self.into()
    }
}

impl From<Term> for Expression {
    fn from(value: Term) -> Self {
        Expression { terms: vec![value] }
    }
}

impl From<usize> for Term {
    fn from(value: usize) -> Self {
        Term::Num(value)
    }
}

impl<E: Into<Expression>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs: Expression = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs
    }
}

impl<E: Into<Expression>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs
    }
}

impl<E: Into<Expression>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs
    }
}

impl<E: Into<Expression>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs
    }
}

impl<E: Into<Expression>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs
    }
}

impl<E: Into<Expression>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs
    }
}

impl<E: Into<Expression>> BitOr<E> for Expression {
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
        let n = Term::Var('x');
        let n = (n.expr() + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])), 768);
    }
}
