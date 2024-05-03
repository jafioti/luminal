use crate::prelude::*;

pub trait PadOfShape<S: Shape> {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)>;
}

impl PadOfShape<R0> for () {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![]
    }
}

impl<A: Dimension, S: Into<Expression>, E: Into<Expression>> PadOfShape<(A,)> for (S, E) {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![(self.0.into(), self.1.into())]
    }
}

impl<A: Dimension, S: Into<Expression>, E: Into<Expression>> PadOfShape<(A,)> for ((S, E),) {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![(self.0 .0.into(), self.0 .1.into())]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        S1: Into<Expression>,
        E1: Into<Expression>,
        S2: Into<Expression>,
        E2: Into<Expression>,
    > PadOfShape<(A, B)> for ((S1, E1), (S2, E2))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0 .0.into(), self.0 .1.into()),
            (self.1 .0.into(), self.1 .1.into()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        S1: Into<Expression>,
        E1: Into<Expression>,
        S2: Into<Expression>,
        E2: Into<Expression>,
        S3: Into<Expression>,
        E3: Into<Expression>,
    > PadOfShape<(A, B, C)> for ((S1, E1), (S2, E2), (S3, E3))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0 .0.into(), self.0 .1.into()),
            (self.1 .0.into(), self.1 .1.into()),
            (self.2 .0.into(), self.2 .1.into()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        S1: Into<Expression>,
        E1: Into<Expression>,
        S2: Into<Expression>,
        E2: Into<Expression>,
        S3: Into<Expression>,
        E3: Into<Expression>,
        S4: Into<Expression>,
        E4: Into<Expression>,
    > PadOfShape<(A, B, C, D)> for ((S1, E1), (S2, E2), (S3, E3), (S4, E4))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0 .0.into(), self.0 .1.into()),
            (self.1 .0.into(), self.1 .1.into()),
            (self.2 .0.into(), self.2 .1.into()),
            (self.3 .0.into(), self.3 .1.into()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        E: Dimension,
        S1: Into<Expression>,
        E1: Into<Expression>,
        S2: Into<Expression>,
        E2: Into<Expression>,
        S3: Into<Expression>,
        E3: Into<Expression>,
        S4: Into<Expression>,
        E4: Into<Expression>,
        S5: Into<Expression>,
        E5: Into<Expression>,
    > PadOfShape<(A, B, C, D, E)> for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0 .0.into(), self.0 .1.into()),
            (self.1 .0.into(), self.1 .1.into()),
            (self.2 .0.into(), self.2 .1.into()),
            (self.3 .0.into(), self.3 .1.into()),
            (self.4 .0.into(), self.4 .1.into()),
        ]
    }
}

impl<
        A: Dimension,
        B: Dimension,
        C: Dimension,
        D: Dimension,
        E: Dimension,
        F: Dimension,
        S1: Into<Expression>,
        E1: Into<Expression>,
        S2: Into<Expression>,
        E2: Into<Expression>,
        S3: Into<Expression>,
        E3: Into<Expression>,
        S4: Into<Expression>,
        E4: Into<Expression>,
        S5: Into<Expression>,
        E5: Into<Expression>,
        S6: Into<Expression>,
        E6: Into<Expression>,
    > PadOfShape<(A, B, C, D, E, F)>
    for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5), (S6, E6))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0 .0.into(), self.0 .1.into()),
            (self.1 .0.into(), self.1 .1.into()),
            (self.2 .0.into(), self.2 .1.into()),
            (self.3 .0.into(), self.3 .1.into()),
            (self.4 .0.into(), self.4 .1.into()),
            (self.5 .0.into(), self.5 .1.into()),
        ]
    }
}

impl<Sh: Shape, S: Into<Expression> + Copy, E: Into<Expression> + Copy> PadOfShape<Sh>
    for &[(S, E)]
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<const N: usize, Sh: Shape, S: Into<Expression> + Copy, E: Into<Expression> + Copy>
    PadOfShape<Sh> for &[(S, E); N]
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<Sh: Shape, S: Into<Expression> + Copy, E: Into<Expression> + Copy> PadOfShape<Sh>
    for &Vec<(S, E)>
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<Sh: Shape, S: Into<Expression>, E: Into<Expression>> PadOfShape<Sh> for Vec<(S, E)> {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.into_iter()
            .map(|(s, e)| (s.into(), e.into()))
            .collect()
    }
}
