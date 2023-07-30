use crate::prelude::*;

pub mod activation;
pub mod embedding;
pub mod linear;
pub mod norm;
pub mod transformer;

pub struct Repeated<T, const N: usize> {
    pub modules: Vec<T>,
}

impl<T: InitModule, const N: usize> InitModule for Repeated<T, N> {
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            modules: (0..N).map(|_| InitModule::initialize(cx)).collect(),
        }
    }
}

impl<T: SerializeModule, const N: usize> SerializeModule for Repeated<T, N> {
    fn serialize(&self, s: &mut Serializer) {
        for (i, l) in self.modules.iter().enumerate() {
            s.module(&format!("layer{i}"), l);
        }
    }
}

impl<I, T: Module<I, Output = I>, const N: usize> Module<I> for Repeated<T, N> {
    type Output = I;

    fn forward(&self, mut input: I) -> Self::Output {
        for m in &self.modules {
            input = m.forward(input);
        }
        input
    }
}

// Tuple impls

impl<X> Module<X> for () {
    type Output = X;
    fn forward(&self, input: X) -> Self::Output {
        input
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),*]) => {
        impl<
            Input,
            $last:
            $(Module::<$rev_tail ::Output>, $rev_tail: )*
            Module<Input>
        > Module<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward(&self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward(x);)+
                x
            }
        }

        impl<$($name: InitModule,)+> InitModule for ($($name,)+) {
            fn initialize(cx: &mut Graph) -> Self {
                (
                $($name::initialize(cx),)+
                )
            }
        }

        impl<$($name: SerializeModule,)+> SerializeModule for ($($name,)+) {
            fn serialize(&self, s: &mut Serializer) {
                $(s.module(&format!("layer{}", $idx), &self.$idx);)+
            }
        }
    };
}

tuple_impls!([M1][0], M1, []);
tuple_impls!([M1, M2] [0, 1], M2, [M1]);
tuple_impls!([M1, M2, M3] [0, 1, 2], M3, [M2, M1]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3], M4, [M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4], M5, [M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5], M6, [M5, M4, M3, M2, M1]);
