use itertools::Itertools;

use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Permute(Ax::as_array().into_iter().map(|i| i as usize).collect_vec()),
                <Dst as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_shape = <Dst as Shape>::realized_shape();
        let mut shape = self.shape_tracker().clone();

        let mut new_id = self.id;
        for (dim, size) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_shape[i as usize]))
        {
            shape.insert(dim, size);
            new_id = graph
                .add_op(op::Expand(dim, size), shape.clone())
                .input(new_id)
                .finish();
        }
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn reshape<N: Shape>(self) -> GraphTensor<N> {
        // <S as AssertSameNumel<N>>::assert_same_numel();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Reshape(<N as Shape>::realized_shape()),
                <N as Shape>::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Dynamically reshape. Panics if destination shape doesn't match given shape.
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<RealDim>) -> GraphTensor<N> {
        for (a, b) in N::realized_shape().iter().zip(shape.iter()) {
            match a {
                RealDim::Const(n) => assert_eq!(RealDim::Const(*n), *b),
                RealDim::Dyn => {}
            }
        }
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Reshape(shape.clone()), shape)
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice<Slice: SliceOfShape<S>>(self, slice: Slice) -> GraphTensor<Slice::OutputShape> {
        let slice = slice.to_range_vec();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(op::Slice(slice), Slice::OutputShape::realized_shape())
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    pub fn pad<Dst: Shape>(self, ranges: Vec<(i32, i32)>) -> GraphTensor<Dst> {
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let new_id = graph
            .add_op(
                op::Function(Box::new(move |inps, _| {
                    let (id, mut st) = (inps[0].1.tensor_id, inps[0].1.shape.clone());
                    st.pad(&ranges);
                    (
                        None,
                        TensorView {
                            tensor_id: id,
                            shape: st,
                        },
                    )
                })),
                Dst::realized_shape(),
            )
            .input(self.id)
            .finish();
        GraphTensor::from_id(new_id, self.graph_ref)
    }

    //     pub fn concat<B, Ax: Axes>(
    //         self,
    //         rhs: GraphTensor<B>,
    //     ) -> GraphTensor<<(A, B) as TryConcatAlong<Ax>>::Output>
    //     where
    //         Dst: Shape<Concrete = S::Concrete> + HasAxes<Ax>,
    //         <(A, B) as TryConcatAlong<Ax>>::Output: Shape,
    //     {
    //     }
    // }
}
