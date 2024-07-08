use egg::*;
use rustc_hash::FxHashMap;
use std::{
    fmt::Debug,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, IndexMut, Mul,
        MulAssign, Rem, RemAssign, Sub, SubAssign,
    },
};
use symbolic_expressions::Sexp;
use tinyvec::ArrayVec;

/// A symbolic expression stored on the stack
pub type Expression = GenericExpression<ArrayVec<[Term; 20]>>; // We need to figure out how to reduce this, can't be fixed at 20. ShapeTracker would take up 6 dims * 12 pads * 12 slices * 20 terms * 8 bytes = 138kb
/// A symbolic expression stored on the heap
pub type BigExpression = GenericExpression<Vec<Term>>;

/// A single term of a symbolic expression such as a variable, number or operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

impl std::fmt::Debug for Term {
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
    pub fn as_op(self) -> Option<fn(i64, i64) -> Option<i64>> {
        match self {
            Term::Add => Some(|a, b| a.checked_add(b)),
            Term::Sub => Some(|a, b| a.checked_sub(b)),
            Term::Mul => Some(|a, b| a.checked_mul(b)),
            Term::Div => Some(|a, b| a.checked_div(b)),
            Term::Mod => Some(|a, b| a.checked_rem(b)),
            Term::Max => Some(|a, b| Some(a.max(b))),
            Term::Min => Some(|a, b| Some(a.min(b))),
            Term::And => Some(|a, b| Some((a != 0 && b != 0) as i64)),
            Term::Or => Some(|a, b| Some((a != 0 || b != 0) as i64)),
            Term::Gte => Some(|a, b| Some((a >= b) as i64)),
            Term::Lt => Some(|a, b| Some((a < b) as i64)),
            _ => None,
        }
    }
}

/// Trait implemented on the 2 main symbolic expression storage types, Vec<Term> and ArrayVec<Term>
#[allow(clippy::len_without_is_empty)]
pub trait ExpressionStorage:
    Clone + IndexMut<usize, Output = Term> + std::iter::Extend<Term> + Default + PartialEq + Debug
{
    fn len(&self) -> usize;
    fn push(&mut self, term: Term);
    fn pop(&mut self) -> Option<Term>;
    fn remove(&mut self, index: usize) -> Term;
    fn into_vec(self) -> Vec<Term>;
    fn iter_ref(&self) -> impl Iterator<Item = &Term>;
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
    fn iter_ref(&self) -> impl Iterator<Item = &Term> {
        self.iter()
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
    fn iter_ref(&self) -> impl Iterator<Item = &Term> {
        self.iter()
    }
}

/// A symbolic expression
#[derive(Clone, Copy, Hash, Eq, serde::Serialize, serde::Deserialize)]
pub struct GenericExpression<S: ExpressionStorage> {
    pub terms: S, // Terms in postfix notation
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
        for term in self.terms.iter_ref() {
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
        if self.terms.len() == 1 {
            return self;
        }
        egg_simplify(self)
    }

    pub fn as_num(&self) -> Option<i32> {
        if let Term::Num(n) = self.terms[0] {
            if self.terms.len() == 1 {
                return Some(n);
            }
        }
        None
    }

    /// Minimum
    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self || rhs == i32::MAX {
            return self;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.min(b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Min);
        rhs
    }

    /// Maximum
    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self || rhs == 0 || self == i32::MAX {
            return self;
        }
        if self == 0 || rhs == i32::MAX {
            return rhs;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.max(b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Max);
        rhs
    }

    /// Greater than or equals
    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return true.into();
        }
        if rhs == i32::MAX {
            return false.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a >= b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Gte);
        rhs
    }

    /// Less than
    pub fn lt<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        if rhs == self {
            return false.into();
        }
        if let Term::Num(n) = rhs.terms[0] {
            if self.terms[self.terms.len() - 1] == Term::Mod && self.terms[0] == Term::Num(n) {
                return true.into();
            }
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a < b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Lt);
        rhs
    }

    /// Substitute an expression for a variable
    pub fn substitute<N: ExpressionStorage>(self, var: char, expr: GenericExpression<N>) -> Self {
        let mut new_terms = S::default();
        for term in self.terms.iter_ref() {
            match term {
                Term::Var(c) if *c == var => {
                    for t in expr.terms.iter_ref() {
                        new_terms.push(*t);
                    }
                }
                _ => {
                    new_terms.push(*term);
                }
            }
        }
        Self { terms: new_terms }
    }
}

impl<S: ExpressionStorage> GenericExpression<S> {
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
        for term in self.terms.iter_ref() {
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
        for term in self.terms.iter_ref() {
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
            .iter_ref()
            .filter_map(|t| match t {
                Term::Var(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    /// Check if the '-' variable exists in the expression.
    pub fn is_unknown(&self) -> bool {
        self.terms.iter_ref().any(|t| matches!(t, Term::Var('-')))
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
        s.extend(value.terms.iter_ref().copied());
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
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a + b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Add);
        rhs
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
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a - b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Sub);
        rhs
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
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            if let Some(c) = a.checked_mul(b) {
                return c.into();
            }
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Mul);
        rhs
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
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a / b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Div);
        rhs
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Rem<E> for GenericExpression<S> {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 || rhs == self {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a % b).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Mod);
        rhs
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
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 && b != 0).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::And);
        rhs
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOr<E> for GenericExpression<S> {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        if rhs == 1 || self == 1 {
            return 1.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 || b != 0).into();
        }
        rhs.terms.extend(self.terms.iter_ref().copied());
        rhs.terms.push(Term::Or);
        rhs
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

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn luminal_to_egg<S: ExpressionStorage>(expr: &GenericExpression<S>) -> RecExpr<Math> {
    let mut stack = Vec::new();

    for term in expr.terms.iter_ref() {
        match term {
            Term::Num(_) | Term::Var(_) => {
                stack.push(symbolic_expressions::Sexp::String(format!("{term:?}")))
            }
            _ => {
                let left = stack.pop().unwrap();
                let right = stack.pop().unwrap();
                let subexpr = symbolic_expressions::Sexp::List(vec![
                    symbolic_expressions::Sexp::String(format!("{term:?}")),
                    left,
                    right,
                ]);
                stack.push(subexpr);
            }
        }
    }
    fn parse_sexp_into<L: FromOp>(
        sexp: &Sexp,
        expr: &mut RecExpr<L>,
    ) -> Result<Id, RecExprParseError<L::Error>> {
        match sexp {
            Sexp::Empty => Err(egg::RecExprParseError::EmptySexp),
            Sexp::String(s) => {
                let node = L::from_op(s, vec![]).map_err(egg::RecExprParseError::BadOp)?;
                Ok(expr.add(node))
            }
            Sexp::List(list) if list.is_empty() => Err(egg::RecExprParseError::EmptySexp),
            Sexp::List(list) => match &list[0] {
                Sexp::Empty => unreachable!("Cannot be in head position"),
                list @ Sexp::List(..) => Err(egg::RecExprParseError::HeadList(list.to_owned())),
                Sexp::String(op) => {
                    let arg_ids: Vec<Id> = list[1..]
                        .iter()
                        .map(|s| parse_sexp_into(s, expr))
                        .collect::<Result<_, _>>()?;
                    let node = L::from_op(op, arg_ids).map_err(egg::RecExprParseError::BadOp)?;
                    Ok(expr.add(node))
                }
            },
        }
    }

    let sexp = stack.pop().unwrap();
    let mut expr = RecExpr::default();
    parse_sexp_into(&sexp, &mut expr).unwrap();
    expr
}

fn egg_to_luminal<S: ExpressionStorage>(expr: RecExpr<Math>) -> GenericExpression<S> {
    fn create_postfix(expr: &[Math]) -> Vec<Term> {
        match expr.last().unwrap() {
            Math::Num(i) => vec![Term::Num(*i)],
            Math::Add([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Add],
            ]
            .concat(),
            Math::Sub([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Sub],
            ]
            .concat(),
            Math::Mul([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Mul],
            ]
            .concat(),
            Math::Div([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Div],
            ]
            .concat(),
            Math::Mod([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Mod],
            ]
            .concat(),
            Math::Min([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Min],
            ]
            .concat(),
            Math::Max([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Max],
            ]
            .concat(),
            Math::And([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::And],
            ]
            .concat(),
            Math::Or([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Or],
            ]
            .concat(),
            Math::LessThan([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Lt],
            ]
            .concat(),
            Math::GreaterThanEqual([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Gte],
            ]
            .concat(),
            Math::Symbol(s) => vec![Term::Var(s.as_str().chars().next().unwrap())],
        }
    }
    let mut terms = S::default();
    terms.extend(create_postfix(expr.as_ref()));
    GenericExpression { terms }
}

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

define_language! {
    enum Math {
        Num(i32),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "min" = Min([Id; 2]),
        "max" = Max([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "<" = LessThan([Id; 2]),
        ">=" = GreaterThanEqual([Id; 2]),
        Symbol(Symbol),
    }
}

#[derive(Default)]
pub struct ConstantFold;
impl Analysis<Math> for ConstantFold {
    type Data = Option<i32>;

    fn make(egraph: &egg::EGraph<Math, ConstantFold>, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| *d);
        Some(match enode {
            Math::Num(c) => *c,
            Math::Add([a, b]) => x(a)?.checked_add(x(b)?)?,
            Math::Sub([a, b]) => x(a)?.checked_sub(x(b)?)?,
            Math::Mul([a, b]) => x(a)?.checked_mul(x(b)?)?,
            Math::Div([a, b]) if x(b) != Some(0) => {
                let (a, b) = (x(a)?, x(b)?);
                if a % b != 0 {
                    return None;
                } else {
                    a.checked_div(b)?
                }
            }
            Math::Mod([a, b]) if x(b) != Some(0) => x(a)?.checked_rem(x(b)?)?,
            Math::Min([a, b]) if x(b) != Some(0) => x(a)?.min(x(b)?),
            Math::Max([a, b]) if x(b) != Some(0) => x(a)?.max(x(b)?),
            Math::And([a, b]) if x(b) != Some(0) => {
                if x(a)? != 0 && x(b)? != 0 {
                    1
                } else {
                    0
                }
            }
            Math::Or([a, b]) if x(b) != Some(0) => {
                if x(a)? != 0 || x(b)? != 0 {
                    1
                } else {
                    0
                }
            }
            Math::LessThan([a, b]) if x(b) != Some(0) => {
                if x(a)? < x(b)? {
                    1
                } else {
                    0
                }
            }
            Math::GreaterThanEqual([a, b]) if x(b) != Some(0) => {
                if x(a)? >= x(b)? {
                    1
                } else {
                    0
                }
            }
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = egraph.add(Math::Num(c));
            egraph.union(id, added);
            // to not prune, comment this out
            egraph[id].nodes.retain(|n| n.is_leaf());
        }
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.map(|i| i != 0).unwrap_or(true)
}

fn make_rules() -> Vec<Rewrite> {
    vec![
        // Communative properties
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("commute-min"; "(min ?a ?b)" => "(min ?b ?a)"),
        rewrite!("commute-max"; "(max ?a ?b)" => "(max ?b ?a)"),
        rewrite!("commute-and"; "(&& ?a ?b)" => "(&& ?b ?a)"),
        rewrite!("commute-or"; "(|| ?a ?b)" => "(|| ?b ?a)"),
        // Associative properties
        rewrite!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        // rewrite!("mul-div-associative"; "(/ (* ?x ?y) ?z)" => "(* ?x (/ ?y ?z))"),
        rewrite!("sub-canon"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"),
        // Simple binary reductions
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
        rewrite!("div-1"; "(/ ?a 1)" => "?a"),
        rewrite!("div-self"; "(/ ?a ?a)" => "1"),
        rewrite!("and-0"; "(&& ?a 0)" => "0"),
        rewrite!("and-1"; "(&& ?a 1)" => "?a"),
        rewrite!("or-0"; "(|| ?a 0)" => "?a"),
        rewrite!("or-1"; "(|| ?a 1)" => "1"),
        rewrite!("min-i32-max"; "(min ?a 2147483647)" => "?a"),
        rewrite!("max-i32-max"; "(max ?a 2147483647)" => "2147483647"),
        rewrite!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero("?x")),
        // rewrite!("add-zero"; "?a" => "(+ ?a 0)"),
        // rewrite!("mul-one";  "?a" => "(* ?a 1)"),
        rewrite!("cancel-sub"; "(- ?a ?a)" => "0"),
        rewrite!("cancel-div"; "(/ ?a ?a)" => "1" if is_not_zero("?a")),
        // Other
        rewrite!("distribute"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("distribute-max"; "(* ?a (max ?b ?c))"        => "(max (* ?a ?b) (* ?a ?c))"),
        rewrite!("distribute-min"; "(* ?a (min ?b ?c))"        => "(min (* ?a ?b) (* ?a ?c))"),
        rewrite!("factor"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rewrite!("group-terms"; "(+ ?a ?a)" => "(* 2 ?a)"),
        rewrite!("distribute-mod"; "(* (% ?b ?c) ?a)" => "(% (* ?b ?a) (* ?c ?a))"),
        rewrite!("explicit-truncate"; "(* (/ ?a ?b) ?b)" => "(- ?a (% ?a ?b))"),
        rewrite!("mul-mod"; "(% (* ?a ?b) ?b)" => "0"),
        // rewrite!("mul-distribute"; "(* ?a (% (/ ?b ?c) ?d))" => "(% (/ ?b (* ?c ?a)) (* ?d ?a))"),
        // rewrite!("div-mod-mul"; "(% (/ ?a ?b) ?c)" => "(% ?a (* ?b ?c))"),
    ]
}

fn egg_simplify<S: ExpressionStorage>(expr: GenericExpression<S>) -> GenericExpression<S> {
    // Convert to egg expression
    let expr = luminal_to_egg(&expr);
    // Simplify
    let runner = Runner::default()
        .with_expr(&expr)
        // .with_iter_limit(100)
        .run(&make_rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);
    // Convert back to luminal expression
    egg_to_luminal(best)
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_expressions() {
        let n = Expression::from('x') + (Expression::from(256) - (Expression::from('x') % 256));
        assert_eq!(
            n.simplify()
                .exec(&[('x', 767)].into_iter().collect())
                .unwrap(),
            768
        );
    }

    #[test]
    fn test_minimizations() {
        let expr = ((BigExpression::from('a') * 1) + 0) / 1 + (1 - 1);
        let reduced_expr = expr.simplify();
        assert_eq!(reduced_expr, 'a');
    }

    #[test]
    fn test_substitution() {
        let main = Expression::from('x') - 255;
        let sub = Expression::from('x') / 2;
        let new = main.substitute('x', sub).simplify();
        assert_eq!(new, (Expression::from('x') / 2) + -255);
    }

    #[test]
    fn test_group_terms() {
        let s = BigExpression::from('s');
        let expr = (s.clone() * ((s.clone() - 4) + 1))
            + (((s.clone() + 1) * ((s.clone() - 4) + 1)) - (s.clone() * ((s.clone() - 4) + 1)));
        assert_eq!(expr.simplify().terms.len(), 7);
    }
}
