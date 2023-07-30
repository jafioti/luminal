use std::any::Any;

use rand::{thread_rng, Rng};

use crate::{op, prelude::*, serialization::SerializeModule};

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
                    let mut shape = tensors[1].shape.shape().clone();
                    shape.push(DIM);
                    Tensor {
                        data: Box::new(res),
                        shape: ShapeTracker::new(shape),
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
        s.weight.set(
            (0..(A * B))
                .map(|_| rng.gen_range(-1_f32..1_f32))
                .collect::<Vec<_>>(),
        );
        s
    }
}

impl<const A: usize, const B: usize> SerializeModule for Embedding<A, B> {
    fn serialize(&self, s: &mut crate::serialization::Serializer) {
        s.tensor("weight", self.weight);
    }
}

// Single
impl<S: Dim, const N: usize, const DIM: usize> Module<GraphTensor<(S,)>> for Embedding<N, DIM> {
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S,)>) -> Self::Output {
        <Self as Module<GraphTensor<(Const<1>, S)>>>::forward(self, input.expand()).reshape()
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

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::Module as DfdxModule,
        tensor::{Cpu, TensorFromVec},
    };

    use crate::{
        prelude::{Module, *},
        tests::assert_close_data,
    };

    use super::Embedding;
    use dfdx::nn::BuildOnDevice;

    #[test]
    fn test_embedding() {
        let mut cx = Graph::new();
        let batch = cx.new_tensor::<R2<2, 3>>();
        let a = cx.new_tensor::<R1<3>>();

        let model: Embedding<3, 4> = InitModule::initialize(&mut cx);
        model
            .weight
            .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        let b = model.forward(a);
        let batch_out = model.forward(batch);

        b.mark();
        a.mark();
        batch_out.mark();
        a.set(vec![1, 0, 1]);
        batch.set(vec![1, 0, 1, 1, 0, 1]);

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
        d_model.weight = d_dev.tensor_from_vec(
            vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
        );
        let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (dfdx::shapes::Const::<3>,));
        let d_batch = d_dev.tensor_from_vec(
            vec![1, 0, 1, 1, 0, 1],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );

        let d_b = d_model.forward(d_a);
        let d_batch_out = d_model.forward(d_batch);

        assert_close_data(&b.retrieve().unwrap().real_data().unwrap(), &d_b.as_vec());
        assert_close_data(
            &batch_out.retrieve().unwrap().real_data().unwrap(),
            &d_batch_out.as_vec(),
        );
    }
}
