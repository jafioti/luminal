use super::Term;

use std::{
    fmt::Debug,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, IndexMut, Mul,
        MulAssign, Rem, RemAssign, Sub, SubAssign,
    },
};

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
    fn into_vec(self) -> Vec<Term>;
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
    fn into_vec(self) -> Vec<Term> {
        self
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
    fn into_vec(self) -> Vec<Term> {
        self.to_vec()
    }
}

/// A symbolic expression
#[derive(Clone, Copy, Hash, Eq)]
pub struct GenericExpression<S: ExpressionStorage> {
    pub terms: S,
}

impl<S: ExpressionStorage, T> PartialEq<T> for GenericExpression<S>
where
    for<'a> &'a T: Into<Self>,
{
    fn eq(&self, other: &T) -> bool {
        self.terms == other.into().terms
    }
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

impl<S: ExpressionStorage + Clone> std::fmt::Display for GenericExpression<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<S: ExpressionStorage> GenericExpression<S> {
    /// Simplify the expression to its minimal terms
    pub fn simplify(self) -> Self {
        crate::simplify::reduce_triples(self)
    }

    /// Minimum
    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return self;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.simplify()
    }

    /// Maximum
    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self || rhs == 0 {
            return self;
        }
        if self == 0 {
            return rhs;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.simplify()
    }

    /// Greater than or equals
    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return 1.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.simplify()
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
        rhs.simplify()
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
        Self { terms: new_terms }.simplify()
    }
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
        self.exec_single_var_stack(value, &mut stack)
    }
    /// Evaluate the expression with one value for all variables. Uses a provided stack
    pub fn exec_single_var_stack(&self, value: usize, stack: &mut Vec<i64>) -> usize {
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Var(_) => stack.push(value as i64),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
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
        stack: &mut Vec<i64>,
    ) -> Option<usize> {
        for term in &self.terms {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as i64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
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

impl Expression {
    pub fn big(&self) -> BigExpression {
        BigExpression::from(*self)
    }
}

impl BigExpression {
    pub fn small(&self) -> Expression {
        Expression::from(self)
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

impl<S: ExpressionStorage, T: ExpressionStorage> From<&GenericExpression<T>>
    for GenericExpression<S>
{
    fn from(value: &GenericExpression<T>) -> Self {
        let mut s = S::default();
        s.extend(value.terms.clone().into_vec());
        Self { terms: s }
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
        if rhs == 0 {
            return self;
        }
        if self == 0 {
            return rhs;
        }
        if self == rhs {
            return self * 2;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Sub<E> for GenericExpression<S> {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 0 {
            return self;
        }
        if self == rhs {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Mul<E> for GenericExpression<S> {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Div<E> for GenericExpression<S> {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == rhs {
            return 1.into();
        }
        if self == 0 {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Rem<E> for GenericExpression<S> {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 || rhs == self {
            return 0.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitAnd<E> for GenericExpression<S> {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOr<E> for GenericExpression<S> {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 || self == 1 {
            return 1.into();
        }
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.simplify()
    }
}

impl<S: ExpressionStorage> std::iter::Product for GenericExpression<S> {
    fn product<I: Iterator<Item = GenericExpression<S>>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p *= n;
        }
        p
    }
}

impl<S: ExpressionStorage, E: Into<Self>> AddAssign<E> for GenericExpression<S> {
    fn add_assign(&mut self, rhs: E) {
        *self = self.clone() + rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> SubAssign<E> for GenericExpression<S> {
    fn sub_assign(&mut self, rhs: E) {
        *self = self.clone() - rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> MulAssign<E> for GenericExpression<S> {
    fn mul_assign(&mut self, rhs: E) {
        *self = self.clone() * rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> DivAssign<E> for GenericExpression<S> {
    fn div_assign(&mut self, rhs: E) {
        *self = self.clone() / rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> RemAssign<E> for GenericExpression<S> {
    fn rem_assign(&mut self, rhs: E) {
        *self = self.clone() % rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitAndAssign<E> for GenericExpression<S> {
    fn bitand_assign(&mut self, rhs: E) {
        *self = self.clone() & rhs;
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOrAssign<E> for GenericExpression<S> {
    fn bitor_assign(&mut self, rhs: E) {
        *self = self.clone() | rhs;
    }
}
