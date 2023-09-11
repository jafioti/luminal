use crate::{op, prelude::*};

impl<S: Shape> GraphTensor<S> {
    pub fn permute<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: PermuteShapeTo<Dst, Ax>,
    {
        self.shape.permute(
            &Ax::as_array()
                .into_iter()
                .map(|i| i as usize)
                .collect::<Vec<_>>(),
        );
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn expand<Dst: Shape, Ax: Axes>(mut self) -> GraphTensor<Dst>
    where
        S: BroadcastShapeTo<Dst, Ax>,
    {
        let new_dims = Dst::realized_shape();
        for (i, dim) in Ax::as_array()
            .into_iter()
            .map(|i| (i as usize, new_dims[i as usize]))
        {
            self.shape.expand(i, dim);
        }

        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn reshape<N: Shape>(self) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, self.shape)
                .finish()
        } else {
            // Already contiguous
            self.id
        };

        GraphTensor::from_id(id, ShapeTracker::new(&N::realized_shape()), self.graph_ref)
    }

    /// Dynamically reshape with annotations for the shape tracker
    pub fn dyn_reshape<N: Shape>(self, shape: Vec<Dim>) -> GraphTensor<N> {
        let id = if !self.shape.is_contiguous() {
            // Insert contiguous call
            self.graph()
                .add_op(op::Contiguous)
                .input(self.id, self.shape)
                .finish()
        } else {
            // Already contiguous
            self.id
        };

        GraphTensor::from_id(id, ShapeTracker::new(&shape), self.graph_ref)
    }

    pub fn realize<Dst: Shape<Concrete = <<S as HasShape>::Shape as Shape>::Concrete>>(
        self,
    ) -> GraphTensor<Dst>
    where
        S: RealizeShapeTo<Dst>,
    {
        GraphTensor::from_id(
            self.id,
            self.shape.realize(&Dst::realized_shape()),
            self.graph_ref,
        )
    }

    pub fn contiguous(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .add_op(op::Contiguous)
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice<Slice: SliceOfShape<S>>(
        mut self,
        slice: Slice,
    ) -> GraphTensor<Slice::OutputShape> {
        self.shape.slice(&slice.to_range_vec());
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn pad<Dst: Shape>(mut self, ranges: Vec<(usize, usize)>) -> GraphTensor<Dst> {
        self.shape.pad(&ranges);
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    pub fn concat_along<Dst: Shape, Ax: Axes<Array = [isize; 1]>, Rhs: Shape>(
        self,
        rhs: GraphTensor<Rhs>,
    ) -> GraphTensor<Dst> {
        let dim = Ax::as_array()[0] as usize;
        let fin = self
            .graph()
            .add_op(op::Function(
                "Concat".to_string(),
                Box::new(move |mut inps| {
                    let mut pad_shape = vec![(0, 0); S::NUM_DIMS];
                    pad_shape[dim].1 = inps[1].1.shape()[dim]
                        .to_usize()
                        .expect("Tried to concat on a dim with an unknown size");
                    inps[0].1.pad(&pad_shape);
                    let mut pad_shape = vec![(0, 0); S::NUM_DIMS];
                    pad_shape[dim].0 = inps[0].1.shape()[dim]
                        .to_usize()
                        .expect("Tried to concat on a dim with an unknown size");
                    inps[1].1.pad(&pad_shape);
                    let (a_data, b_data) = (
                        inps[0]
                            .0
                            .borrowed()
                            .data
                            .as_any()
                            .downcast_ref::<Vec<f32>>()
                            .unwrap(),
                        inps[1]
                            .0
                            .borrowed()
                            .data
                            .as_any()
                            .downcast_ref::<Vec<f32>>()
                            .unwrap(),
                    );
                    // Each input is already padded, so the final answer will be the same size as each input
                    let mut data = vec![0.; inps[0].1.n_elements()];
                    for (i, d) in data.iter_mut().enumerate() {
                        *d = inps[0]
                            .1
                            .index(i)
                            .map(|i| {
                                println!("A: {i}");
                                a_data[i]
                            })
                            .unwrap_or_default()
                            + inps[1].1.index(i).map(|i| b_data[i]).unwrap_or_default()
                    }
                    Tensor {
                        data: Box::new(data),
                    }
                }),
            ))
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        GraphTensor::from_id(
            fin,
            ShapeTracker::new(&Dst::realized_shape()),
            self.graph_ref,
        )
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        shapes::Rank2,
        tensor::{Cpu, TensorFrom, TensorFromVec},
        tensor_ops::{RealizeTo, TryConcatAlong},
    };

    use crate::{prelude::*, tests::assert_close_data};

    #[test]
    fn test_concat_1d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R1<4>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = cx.new_tensor::<R1<3>>("Input");
        b.set(vec![2.30434, 2.2343113, 1.4393]);
        let c = a.concat_along::<R1<7>, crate::prelude::Axis<0>, _>(b);
        c.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b = d_dev.tensor([2.30434, 2.2343113, 1.4393]);
        let d_c = (d_a.realize::<(usize,)>(), d_b.realize::<(usize,)>())
            .concat_along(dfdx::shapes::Axis::<0>);

        assert_close_data(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_concat_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<3, 2>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = cx.new_tensor::<R2<3, 2>>("Input");
        b.set(vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054]);
        let c = a.concat_along::<R2<3, 4>, Axis<1>, _>(b);
        // let d = a.concat_along::<R2<6, 2>, Axis<0>, _>(b);
        c.mark();
        // d.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_c = (
            d_a.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>);
        let d_d = (
            d_a.realize::<(usize, dfdx::shapes::Const<2>)>(),
            d_b.realize::<(usize, dfdx::shapes::Const<2>)>(),
        )
            .concat_along(dfdx::shapes::Axis::<0>);

        println!("Executed");
        let c = c.data();
        println!("C: {:?}", c);
        println!("D: {:?}", d_c.as_vec());
        assert_close_data(&c, &d_c.as_vec());
        // assert_close_data(&d.data(), &d_d.as_vec());
    }

    #[test]
    fn test_pad_2d() {
        let mut cx = Graph::new();
        let a = cx.new_tensor::<R2<3, 2>>("Input");
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.pad::<R2<3, 4>>(vec![(0, 0), (0, 2)]);
        b.mark();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![0., 0., 0., 0., 0., 0.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = (
            d_a.realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>)
            .realize::<Rank2<3, 4>>();

        println!("B: {:?}", b.shape);
        let b = b.data();
        println!("C: {:?}", b);
        println!("D: {:?}", d_b.as_vec());
        assert_close_data(&b, &d_b.as_vec());
    }
}
