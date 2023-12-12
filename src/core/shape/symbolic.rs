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
        let mut symbols = vec![];
        for term in self.terms {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => c.to_string(),
                Term::Max => format!(
                    "max({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        write!(f, "{}", symbols.pop().unwrap())
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
            match (self.terms[i], self.terms[i + 1], self.terms[i + 2]) {
                (Term::Num(b), Term::Num(a), term) if term.as_op().is_some() => {
                    self.terms[i] = Term::Num(term.as_op().unwrap()(a, b));
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove min(i, inf) and min(inf, i)
                (Term::Num(b), Term::Num(_) | Term::Var(_), Term::Min)
                    if b == i32::MAX as usize =>
                {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(a), Term::Min) if a == i32::MAX as usize => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove max(i, 0) and max(0, i)
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(0), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i + 0, i - 0 and 0 + i
                (_, Term::Num(0), Term::Add) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Add | Term::Sub) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 0, 0 * i
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove 0 / i
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 1 and 1 * i
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(1), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i / 1
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
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
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.minimize()
    }

    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.minimize()
    }

    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.minimize()
    }

    pub fn lt<E: Into<Self>>(self, rhs: E) -> Self {
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
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.minimize()
    }

    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.minimize()
    }

    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.minimize()
    }

    pub fn lt<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs.minimize()
    }

    pub fn minimize(mut self) -> Self {
        let mut i = 0;
        while i < self.terms.len().saturating_sub(2) {
            match (self.terms[i], self.terms[i + 1], self.terms[i + 2]) {
                (Term::Num(b), Term::Num(a), term) if term.as_op().is_some() => {
                    self.terms[i] = Term::Num(term.as_op().unwrap()(a, b));
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove min(i, inf) and min(inf, i)
                (Term::Num(b), Term::Num(_) | Term::Var(_), Term::Min)
                    if b == i32::MAX as usize =>
                {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(a), Term::Min) if a == i32::MAX as usize => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove max(i, 0) and max(0, i)
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(0), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i + 0, i - 0 and 0 + i
                (_, Term::Num(0), Term::Add) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Add | Term::Sub) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 0, 0 * i
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove 0 / i
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 1 and 1 * i
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(1), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i / 1
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                _ => {
                    i += 1;
                }
            }
        }

        self
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
            Term::Sub => Some(|a, b| a.saturating_sub(b)),
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

impl From<char> for Expression {
    fn from(value: char) -> Self {
        Expression::from(Term::Var(value))
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::from(Term::Num(value))
    }
}

impl From<&usize> for Expression {
    fn from(value: &usize) -> Self {
        Expression::from(Term::Num(*value))
    }
}

impl From<i32> for Expression {
    fn from(value: i32) -> Self {
        Expression::from(value as usize)
    }
}

impl From<char> for BigExpression {
    fn from(value: char) -> Self {
        BigExpression::from(Term::Var(value))
    }
}

impl From<usize> for BigExpression {
    fn from(value: usize) -> Self {
        BigExpression::from(Term::Num(value))
    }
}

impl From<&usize> for BigExpression {
    fn from(value: &usize) -> Self {
        BigExpression::from(Term::Num(*value))
    }
}

impl From<i32> for BigExpression {
    fn from(value: i32) -> Self {
        BigExpression::from(value as usize)
    }
}

impl From<Expression> for BigExpression {
    fn from(value: Expression) -> Self {
        Self {
            terms: value.terms.to_vec(),
        }
    }
}

impl<E: Into<Self>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<E: Into<Self>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<E: Into<Self>> BitOr<E> for Expression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Add<E> for BigExpression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Sub<E> for BigExpression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Mul<E> for BigExpression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Div<E> for BigExpression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<E: Into<Self>> Rem<E> for BigExpression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<E: Into<Self>> BitAnd<E> for BigExpression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<E: Into<Self>> BitOr<E> for BigExpression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.minimize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);

        let n = (BigExpression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);
    }
}
