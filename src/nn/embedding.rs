use std::any::Any;

use rand::{thread_rng, Rng};

use crate::{op, prelude::*};

// This is a really shoddy way to do gathers TODO: Do something else

impl Data for Vec<usize> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Gather batch of batches from 2D embedding matrix
impl<S: Dim, const DIM: usize> GraphTensor<(S, Const<DIM>)> {
    pub fn gather<S1: Dim, S2: Dim>(
        self,
        indexes: GraphTensor<(S1, S2)>,
    ) -> GraphTensor<(S1, S2, Const<DIM>)> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let res = graph
            .add_op(
                op::Function(Box::new(|tensors| {
                    let data = tensors[0].data.as_any().downcast_ref::<Vec<f32>>().unwrap();
                    let data_idx_fn = tensors[0].shape.index_fn();
                    let indexes = tensors[1]
                        .data
                        .as_any()
                        .downcast_ref::<Vec<usize>>()
                        .unwrap();
                    let index_idx_fn = tensors[1].shape.index_fn();
                    let mut res = Vec::with_capacity(indexes.len() * DIM);
                    for i in 0..indexes.len() {
                        let start = indexes[(index_idx_fn)(i)] * DIM;
                        for n in 0..DIM {
                            res.push(data[(data_idx_fn)(start + n)]);
                        }
                    }
                    Tensor {
                        data: Box::new(res),
                        shape: ShapeTracker::new(vec![indexes.len(), DIM]),
                    }
                })),
                vec![S1::const_size(), RealDim::Const(DIM)],
            )
            .input(self.id)
            .input(indexes.id)
            .finish();

        GraphTensor::from_id(res, self.graph_ref)
    }
}

pub struct Embedding<const N: usize, const DIM: usize> {
    weight: GraphTensor<R2<N, DIM>>,
}

impl<const A: usize, const B: usize> InitModule for Embedding<A, B> {
    fn initialize(cx: &mut Graph) -> Self {
        let s = Self {
            weight: cx.new_tensor(),
        };
        // Init weight as uniform(-1, 1)
        let mut rng = thread_rng();
        s.weight
            .set((0..(A * B)).map(|_| rng.gen_range(-1_f32..1_f32)).collect());
        s
    }
}

// Single
impl<S: Dim, const N: usize, const DIM: usize> Module<GraphTensor<(S,)>> for Embedding<N, DIM> {
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S,)>) -> Self::Output {
        <Self as Module<GraphTensor<(Const<1>, S)>>>::forward(self, input.expand()).max_reduce()
    }
}

// Batch
impl<B: Dim, S: Dim, const N: usize, const DIM: usize> Module<GraphTensor<(B, S)>>
    for Embedding<N, DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S)>) -> Self::Output {
        self.weight.gather(input)
    }
}
