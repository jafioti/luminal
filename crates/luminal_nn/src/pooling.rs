use luminal::prelude::*;

pub struct AvgPool2D {
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl AvgPool2D {
    pub fn new(kernel: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel, stride }
    }
}

impl SerializeModule for AvgPool2D {
    fn serialize(&self, _s: &mut luminal::module::Serializer) {
        // No parameters to serialize for average pooling
    }
}

impl AvgPool2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        // Input: (batch (optional), ch_in, dimx_in, dimy_in)
        let mut expanded = false;
        if input.shape.len() == 3 {
            // Expand batch
            input = input.expand(0, 1);
            expanded = true;
        }
        let (batch, ch_in, dimx_in, dimy_in) = input.dims4();
        let dimx_out = ((dimx_in - self.kernel.0) / self.stride.0 + 1).simplify();
        let dimy_out = ((dimy_in - self.kernel.1) / self.stride.1 + 1).simplify();

        let output = input
            .pool_last_dim(self.kernel.1, self.stride.1, 1) // dilation = 1 for pooling
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(self.kernel.0, self.stride.0, 1)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((
                batch,
                ch_in,
                self.kernel.0 * self.kernel.1,
                dimx_out * dimy_out,
            ))
            .mean_reduce(2) // Average over the kernel dimension
            .reshape((batch, ch_in, dimx_out, dimy_out));

        if expanded {
            output.reshape((ch_in, dimx_out, dimy_out))
        } else {
            output
        }
    }
}

pub struct AdaptiveAvgPool2D {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl SerializeModule for AdaptiveAvgPool2D {
    fn serialize(&self, _s: &mut luminal::module::Serializer) {
        // No learnable parameters
    }
}

impl AdaptiveAvgPool2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        let mut expanded = false;
        // Handle missing batch dimension
        if input.shape.len() == 3 {
            input = input.expand(0, 1);
            expanded = true;
        }

        // Extract dimensions
        let (batch, ch, h_in, w_in) = input.dims4();
        let (h_out, w_out) = self.output_size;

        let stride_h = (h_in / h_out).simplify();
        let stride_w = (w_in / w_out).simplify();
        let kernel_h = (h_in - (h_out - 1) * stride_h).simplify();
        let kernel_w = (w_in - (w_out - 1) * stride_w).simplify();

        // Two-stage pooling (Y then X), followed by averaging over the kernel window
        let mut output = input
            .pool_last_dim(kernel_w, stride_w, 1)
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(kernel_h, stride_h, 1)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((batch, ch, kernel_h * kernel_w, h_out * w_out))
            .mean_reduce(2)
            .reshape((batch, ch, h_out, w_out));

        // Remove batch dim if it was originally absent
        if expanded {
            output = output.reshape((ch, h_out, w_out));
        }

        output
    }
}
