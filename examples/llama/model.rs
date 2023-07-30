use luminal::prelude::*;

// Full LLaMa model implementation, heavily based off of https://github.com/coreylowman/llama-dfdx/blob/main/src/modeling.rs

pub struct Mlp<const I: usize, const H: usize> {
    pub gate_proj: GraphTensor<R2<I, H>>,
    pub down_proj: GraphTensor<R2<H, I>>,
    pub up_proj: GraphTensor<R2<I, H>>,
}

impl<const I: usize, const H: usize, B: Dim, S: Dim> Module<GraphTensor<(B, S, Const<H>)>>
    for Mlp<I, H>
{
    type Output = GraphTensor<(B, S, Const<H>)>;

    fn forward(&self, input: GraphTensor<(B, S, Const<H>)>) -> Self::Output {
        let gate = {
            let gate = input.matmul(self.gate_proj.permute());
            gate.sigmoid() * gate
        };
        let up = {
            let up = input.matmul(self.up_proj.permute());
            up * gate
        };
        up.matmul(self.down_proj.permute())
    }
}
