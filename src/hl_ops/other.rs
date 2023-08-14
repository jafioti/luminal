use crate::prelude::*;

impl<D: Dim> GraphTensor<(D,)> {
    pub fn cumsum_1d(self) -> GraphTensor<(D,)> {
        let x = self.expand::<(Const<1>, Const<1>, Const<1>, D), _>();
        let graph = unsafe { self.graph_ref.as_mut().unwrap() };
        let conv_weight = graph.new_tensor::<R0>("Conv Weight");
        conv_weight.set(vec![1.]);
        let conv_weight = conv_weight.expand::<(Const<1>, Const<1>, Const<1>, D), _>();
        todo!()
    }
}
