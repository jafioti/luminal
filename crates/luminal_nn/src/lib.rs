use luminal::prelude::*;

mod activation;
pub use activation::*;
mod convolution;
pub use convolution::*;
mod embedding;
pub use embedding::*;
mod linear;
pub use linear::*;
mod norm;
pub use norm::*;
mod transformer;
pub use transformer::*;

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
