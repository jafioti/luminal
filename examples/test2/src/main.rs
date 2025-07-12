use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    // Make an input tensor
    let mut a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    let mut b = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    let mut c = a * b;
    let mut d = c + a;
    let f = d.sin();
    let mut e = f.retrieve();
    // Feed tensor through model

    // Display the graph to see the ops
    cx.display();
    // Execute the graph
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            (
                luminal_metal::MetalCompilerPreBuffer::<f16>::default(),
                luminal_metal::quantized::MetalQuantizedCompiler::<f16>::new(q_weights),
                luminal_metal::BufferCompilers::default(),
            ),
            #[cfg(feature = "cuda")]
            (
                luminal_cuda::CudaCompiler::<f16>::default(),
                luminal_cuda::CudaQuantizedCompiler::<f16>::new(q_weights),
            ),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
        ),
        (&mut a, &mut b, &mut c, &mut e),
    );
    cx.execute_debug();
    // Print the results
    println!("B: {:?}", e.data());
}
