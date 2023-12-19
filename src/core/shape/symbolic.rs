#![allow(private_bounds)]
// Super minimal symbolic algebra library

use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, BitAnd, BitOr, Div, IndexMut, Mul, Rem, Sub},
};

use itertools::Itertools;
use tinyvec::ArrayVec;

/// A symbolic expression stored on the stack
pub type Expression = GenericExpression<ArrayVec<[Term; 20]>>; // We need to figure out how to reduce this, can't be fixed at 20. ShapeTracker would take up 6 dims * 12 pads * 12 slices * 20 terms * 8 bytes = 138kb
/// A symbolic expression stored on the heap
pub type BigExpression = GenericExpression<Vec<Term>>;

/// Trait implemented on the 2 main symbolic expression storage types, Vec<Term> and ArrayVec<Term>
#[allow(clippy::len_without_is_empty)]
trait ExpressionStorage:
    Clone
    + IndexMut<usize, Output = Term>
    + std::iter::Extend<Term>
    + IntoIterator<Item = Term>
    + Default
{
    fn len(&self) -> usize;
    fn push(&mut self, term: Term);
    fn pop(&mut self) -> Option<Term>;
    fn remove(&mut self, index: usize) -> Term;
}

// Implement the main storage types
impl ExpressionStorage for Vec<Term> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn push(&mut self, term: Term) {
        Vec::push(self, term)
    }
    fn pop(&mut self) -> Option<Term> {
        Vec::pop(self)
    }
    fn remove(&mut self, index: usize) -> Term {
        Vec::remove(self, index)
    }
}

impl<const C: usize> ExpressionStorage for ArrayVec<[Term; C]>
where
    [Term; C]: tinyvec::Array<Item = Term>,
{
    fn len(&self) -> usize {
        ArrayVec::len(self)
    }
    fn push(&mut self, term: Term) {
        ArrayVec::push(self, term)
    }
    fn pop(&mut self) -> Option<Term> {
        ArrayVec::pop(self)
    }
    fn remove(&mut self, index: usize) -> Term {
        ArrayVec::remove(self, index)
    }
}

/// A symbolic expression
#[derive(Clone, Default)]
pub struct GenericExpression<S: ExpressionStorage> {
    pub terms: S,
}

impl<S: Copy + ExpressionStorage> Copy for GenericExpression<S> {}

impl<S: PartialEq + ExpressionStorage> PartialEq for GenericExpression<S> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl<S: ExpressionStorage + Clone> Debug for GenericExpression<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut symbols = vec![];
        for term in self.terms.clone() {
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

impl<S: ExpressionStorage> GenericExpression<S> {
    pub fn minimize(mut self) -> Self {
        fn get_triples<S: ExpressionStorage>(
            exp: &GenericExpression<S>,
        ) -> Vec<(Option<usize>, usize, Option<usize>)> {
            // Mark all terms with their index
            let terms = exp
                .terms
                .clone()
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            let mut stack = Vec::new();
            let mut triples = vec![];
            for (index, term) in terms {
                match term {
                    Term::Num(_) | Term::Var(_) => stack.push((Some(index), term)),
                    _ => {
                        let (a_ind, a_term) = stack.pop().unwrap();
                        let (b_ind, b_term) = stack.pop().unwrap();
                        triples.push((a_ind, index, b_ind));
                        if let (Term::Num(a), Term::Num(b)) = (a_term, b_term) {
                            stack.push((None, Term::Num(term.as_op().unwrap()(a, b))));
                        } else if let Term::Var(a) = a_term {
                            stack.push((None, Term::Var(a)));
                        } else if let Term::Var(b) = b_term {
                            stack.push((None, Term::Var(b)));
                        }
                    }
                }
            }
            triples
        }
        fn remove_terms<S: ExpressionStorage>(terms: &mut S, inds: &[usize]) {
            for ind in inds.iter().sorted().rev() {
                terms.remove(*ind);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            let triples = get_triples(&self);
            for (a_ind, op_ind, b_ind) in triples {
                let mut inner_changed = true;
                match (
                    a_ind.map(|a| self.terms[a]),
                    self.terms[op_ind],
                    b_ind.map(|b| self.terms[b]),
                ) {
                    (Some(Term::Num(a)), term, Some(Term::Num(b))) if term.as_op().is_some() => {
                        self.terms[a_ind.unwrap()] = Term::Num(term.as_op().unwrap()(a, b));
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Remove min(i, inf) and min(inf, i)
                    (Some(Term::Num(a)), Term::Min, _) if a == i32::MAX as usize => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (_, Term::Min, Some(Term::Num(b))) if b == i32::MAX as usize => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Remove min(i, 0) and min(0, i)
                    (Some(Term::Num(0)), Term::Min, _) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    (_, Term::Min, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    // Remove max(i, 0) and max(0, i)
                    (_, Term::Max, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    (Some(Term::Num(0)), Term::Max, _) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    // Remove max(i, inf) and max(inf, i)
                    (_, Term::Max, Some(Term::Num(i))) if i == i32::MAX as usize => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (Some(Term::Num(i)), Term::Max, _) if i == i32::MAX as usize => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Remove i + 0, i - 0 and 0 + i
                    (Some(Term::Num(0)), Term::Add, _) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (_, Term::Add | Term::Sub, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()])
                    }
                    // Simplify i * 0, 0 * i to 0
                    (_, Term::Mul, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (Some(Term::Num(0)), Term::Mul, _) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify 0 / i to 0
                    (Some(Term::Num(0)), Term::Div, _) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Remove i / 1
                    (_, Term::Div, Some(Term::Num(1))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Remove i * 1 and 1 * i
                    (_, Term::Mul, Some(Term::Num(1))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    (Some(Term::Num(1)), Term::Mul, _) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    // Simplify i - i to 0
                    (Some(a), Term::Sub, Some(b)) if a == b => {
                        self.terms[a_ind.unwrap()] = Term::Num(0);
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify true && i and i && true to i
                    (_, Term::And, Some(Term::Num(1))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    (Some(Term::Num(1)), Term::And, _) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    // Simplify false && i and i && false to false
                    (_, Term::And, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (Some(Term::Num(0)), Term::And, _) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify false || i and i || false to i
                    (_, Term::Or, Some(Term::Num(0))) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    (Some(Term::Num(0)), Term::Or, _) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    // Simplify true || i and i || true to true
                    (_, Term::Or, Some(Term::Num(1))) => {
                        remove_terms(&mut self.terms, &[op_ind, a_ind.unwrap()]);
                    }
                    (Some(Term::Num(1)), Term::Or, _) => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify i >= i to true (1)
                    (Some(a), Term::Gte, Some(b)) if a == b => {
                        self.terms[a_ind.unwrap()] = Term::Num(1);
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify i < i to false (0)
                    (Some(a), Term::Lt, Some(b)) if a == b => {
                        self.terms[a_ind.unwrap()] = Term::Num(0);
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    // Simplify min(i, i) and max(i, i) to i
                    (Some(a), Term::Min | Term::Max, Some(b)) if a == b => {
                        remove_terms(&mut self.terms, &[op_ind, b_ind.unwrap()]);
                    }
                    _ => {
                        inner_changed = false;
                    }
                }
                if inner_changed {
                    changed = true;
                    break;
                }
            }
        }
        self
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
}

impl<S: ExpressionStorage + Clone> GenericExpression<S> {
    /// Evaluate the expression with no variables. Returns Some(value) if no variables are required, otherwise returns None.
    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&HashMap::default())
    }
    /// Evaluate the expression with one value for all variables.
    pub fn exec_single_var(&self, value: usize) -> usize {
        let mut stack = Vec::new();
        for term in self.terms.clone() {
            match term {
                Term::Num(n) => stack.push(n),
                Term::Var(_) => stack.push(value),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop().unwrap()
    }
    /// Evaluate the expression given variables.
    pub fn exec(&self, variables: &HashMap<char, usize>) -> Option<usize> {
        let mut stack = Vec::new();
        for term in self.terms.clone() {
            match term {
                Term::Num(n) => stack.push(n),
                Term::Var(c) => {
                    if let Some(n) = variables.get(&c) {
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
    /// Retrieve all symbols in the expression.
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .clone()
            .into_iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// Check if the '-' variable exists in the expression.
    pub fn is_unknown(&self) -> bool {
        self.terms
            .clone()
            .into_iter()
            .any(|t| matches!(t, Term::Var('-')))
    }
}

/// A single term of a symbolic expression such as a variable, number or operation.
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

impl<S: ExpressionStorage> From<Term> for GenericExpression<S> {
    fn from(value: Term) -> Self {
        let mut terms = S::default();
        terms.push(value);
        GenericExpression { terms }
    }
}

impl<S: ExpressionStorage> From<char> for GenericExpression<S> {
    fn from(value: char) -> Self {
        GenericExpression::from(Term::Var(value))
    }
}

impl<S: ExpressionStorage> From<&char> for GenericExpression<S> {
    fn from(value: &char) -> Self {
        GenericExpression::from(Term::Var(*value))
    }
}

impl<S: ExpressionStorage> From<usize> for GenericExpression<S> {
    fn from(value: usize) -> Self {
        GenericExpression::from(Term::Num(value))
    }
}

impl<S: ExpressionStorage> From<&usize> for GenericExpression<S> {
    fn from(value: &usize) -> Self {
        GenericExpression::from(Term::Num(*value))
    }
}

impl<S: ExpressionStorage> From<i32> for GenericExpression<S> {
    fn from(value: i32) -> Self {
        GenericExpression::from(value as usize)
    }
}

impl<S: ExpressionStorage> From<&i32> for GenericExpression<S> {
    fn from(value: &i32) -> Self {
        GenericExpression::from(*value as usize)
    }
}

impl From<Expression> for BigExpression {
    fn from(value: Expression) -> Self {
        Self {
            terms: value.terms.to_vec(),
        }
    }
}

impl From<BigExpression> for Expression {
    fn from(value: BigExpression) -> Self {
        let mut terms = ArrayVec::new();
        terms.extend(value.terms);
        Self { terms }
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Add<E> for GenericExpression<S> {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Sub<E> for GenericExpression<S> {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Mul<E> for GenericExpression<S> {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Div<E> for GenericExpression<S> {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Rem<E> for GenericExpression<S> {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitAnd<E> for GenericExpression<S> {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOr<E> for GenericExpression<S> {
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

        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);
    }
}
