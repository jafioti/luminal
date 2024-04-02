#![allow(private_bounds)]
// Super minimal symbolic algebra library

use std::{
    fmt::Debug,
    ops::{Add, BitAnd, BitOr, Div, IndexMut, Mul, Rem, Sub},
};

use itertools::Itertools;
use rustc_hash::FxHashMap;
use tinyvec::ArrayVec;

/// A symbolic expression stored on the stack
pub type Expression = GenericExpression<ArrayVec<[Term; 20]>>; // We need to figure out how to reduce this, can't be fixed at 20. ShapeTracker would take up 6 dims * 12 pads * 12 slices * 20 terms * 8 bytes = 138kb
/// A symbolic expression stored on the heap
pub type BigExpression = GenericExpression<Vec<Term>>;

/// Trait implemented on the 2 main symbolic expression storage types, Vec<Term> and ArrayVec<Term>
#[allow(clippy::len_without_is_empty)]
pub trait ExpressionStorage:
    Clone
    + IndexMut<usize, Output = Term>
    + std::iter::Extend<Term>
    + IntoIterator<Item = Term>
    + Default
    + PartialEq
    + Debug
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
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct GenericExpression<S: ExpressionStorage> {
    pub terms: S,
}

impl<S: ExpressionStorage> Default for GenericExpression<S> {
    fn default() -> Self {
        let mut s = S::default();
        s.push(Term::Num(0));
        Self { terms: s }
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
    /// Reduce the expression to its minimal terms
    pub fn minimize(self) -> Self {
        reduce_triples(self)
    }

    /// Minimum
    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return self;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.minimize()
    }

    /// Maximum
    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return self;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.minimize()
    }

    /// Greater than or equals
    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return 1.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.minimize()
    }

    /// Less than
    pub fn lt<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return 0.into();
        }
        if let Term::Num(n) = rhs.terms[0] {
            if self.terms[self.terms.len() - 1] == Term::Mod && self.terms[0] == Term::Num(n) {
                return 1.into();
            }
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs.minimize()
    }

    /// Substitute an expression for a variable
    pub fn substitute<N: ExpressionStorage>(self, var: char, expr: GenericExpression<N>) -> Self {
        let mut new_terms = S::default();
        for term in self.terms.clone().into_iter() {
            match term {
                Term::Var(c) if c == var => {
                    for t in expr.terms.clone().into_iter() {
                        new_terms.push(t);
                    }
                }
                _ => {
                    new_terms.push(term);
                }
            }
        }
        Self { terms: new_terms }.minimize()
    }
}

fn reduce_triples<S: ExpressionStorage>(mut expr: GenericExpression<S>) -> GenericExpression<S> {
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

    #[macro_export]
    macro_rules! unwrap_cont {
        ($i: expr) => {
            if let Some(s) = $i {
                s
            } else {
                continue;
            }
        };
    }
    let mut changed = true;
    while changed {
        changed = false;
        let triples = get_triples(&expr);
        for (a_ind, op_ind, b_ind) in triples {
            let mut inner_changed = true;
            match (
                a_ind.map(|a| expr.terms[a]),
                expr.terms[op_ind],
                b_ind.map(|b| expr.terms[b]),
            ) {
                (Some(Term::Num(a)), term, Some(Term::Num(b))) if term.as_op().is_some() => {
                    expr.terms[unwrap_cont!(a_ind)] = Term::Num(term.as_op().unwrap()(a, b));
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
                }
                // Remove min(i, inf) and min(inf, i)
                (Some(Term::Num(a)), Term::Min, _) if a == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(a_ind)]);
                }
                (_, Term::Min, Some(Term::Num(b))) if b == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
                }
                // Remove max(i, inf) and max(inf, i)
                (_, Term::Max, Some(Term::Num(i))) if i == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(a_ind)]);
                }
                (Some(Term::Num(i)), Term::Max, _) if i == i32::MAX => {
                    remove_terms(&mut expr.terms, &[op_ind, unwrap_cont!(b_ind)]);
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
    expr
}

impl<S: ExpressionStorage> GenericExpression<S>
where
    for<'a> &'a S: IntoIterator<Item = &'a Term>,
{
    /// Evaluate the expression with no variables. Returns Some(value) if no variables are required, otherwise returns None.
    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&FxHashMap::default())
    }
    /// Evaluate the expression with one value for all variables.
    pub fn exec_single_var(&self, value: usize) -> usize {
        let mut stack = Vec::new();
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n),
                Term::Var(_) => stack.push(value as i32),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop().unwrap() as usize
    }
    /// Evaluate the expression given variables.
    pub fn exec(&self, variables: &FxHashMap<char, usize>) -> Option<usize> {
        self.exec_stack(variables, &mut Vec::new())
    }
    /// Evaluate the expression given variables. This function requires a stack to be given for use as storage
    pub fn exec_stack(
        &self,
        variables: &FxHashMap<char, usize>,
        stack: &mut Vec<i32>,
    ) -> Option<usize> {
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as i32)
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
        stack.pop().map(|i| i as usize)
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
    Num(i32),
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
    pub fn as_op(self) -> Option<fn(i32, i32) -> i32> {
        match self {
            Term::Add => Some(std::ops::Add::add),
            Term::Sub => Some(std::ops::Sub::sub),
            Term::Mul => Some(std::ops::Mul::mul),
            Term::Div => Some(std::ops::Div::div),
            Term::Mod => Some(std::ops::Rem::rem),
            Term::Max => Some(core::cmp::Ord::max),
            Term::Min => Some(core::cmp::Ord::min),
            Term::And => Some(|a, b| (a != 0 && b != 0) as i32),
            Term::Or => Some(|a, b| (a != 0 || b != 0) as i32),
            Term::Gte => Some(|a, b| (a >= b) as i32),
            Term::Lt => Some(|a, b| (a < b) as i32),
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
        GenericExpression::from(Term::Num(value as i32))
    }
}

impl<S: ExpressionStorage> From<&usize> for GenericExpression<S> {
    fn from(value: &usize) -> Self {
        GenericExpression::from(Term::Num(*value as i32))
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

impl<S: ExpressionStorage> From<bool> for GenericExpression<S> {
    fn from(value: bool) -> Self {
        GenericExpression::from(value as usize)
    }
}

impl<S: ExpressionStorage> From<&bool> for GenericExpression<S> {
    fn from(value: &bool) -> Self {
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

impl From<&Expression> for Expression {
    fn from(value: &Expression) -> Self {
        *value
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
        if rhs == Self::from(0) {
            return self;
        }
        if self == Self::from(0) {
            return rhs;
        }
        if self == rhs {
            return self * 2;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Sub<E> for GenericExpression<S> {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(0) {
            return self;
        }
        if self == rhs {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Mul<E> for GenericExpression<S> {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(1) {
            return self;
        }
        if self == Self::from(1) {
            return rhs;
        }
        if rhs == Self::from(0) || self == Self::from(0) {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Div<E> for GenericExpression<S> {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(1) {
            return self;
        }
        if self == rhs {
            return 1.into();
        }
        if self == Self::from(0) {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Rem<E> for GenericExpression<S> {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(1) || rhs == self {
            return Self::from(0);
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitAnd<E> for GenericExpression<S> {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(0) || self == Self::from(0) {
            return Self::from(0);
        }
        if rhs == Self::from(1) {
            return self;
        }
        if self == Self::from(1) {
            return rhs;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOr<E> for GenericExpression<S> {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == Self::from(1) || self == Self::from(1) {
            return 1.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage> std::iter::Product for GenericExpression<S> {
    fn product<I: Iterator<Item = GenericExpression<S>>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p = p * n;
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&[('x', 767)].into_iter().collect()).unwrap(), 768);

        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&[('x', 767)].into_iter().collect()).unwrap(), 768);
    }

    #[test]
    fn test_minimizations() {
        let expr = ((BigExpression::from('a') * 1) + 0) / 1 + (1 - 1);
        let reduced_expr = expr.minimize();
        assert_eq!(reduced_expr, 'a'.into());
    }

    #[test]
    fn test_substitution() {
        let main = Expression::from('x') - 255;
        let sub = Expression::from('x') / 2;
        let new = main.substitute('x', sub);
        assert_eq!(new, (Expression::from('x') / 2) - 255);
    }
}
