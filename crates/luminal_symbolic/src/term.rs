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
