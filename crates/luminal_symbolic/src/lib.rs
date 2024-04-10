// Super minimal symbolic algebra library

mod term;
pub use term::Term;
mod expression;
pub use expression::*;
mod simplify;

#[cfg(test)]
mod tests;
