# luminal
![image](https://raw.githubusercontent.com/jafioti/luminal/main/resources/dag.jpeg)
**Deep learning at the speed of light.**

Luminal is a deep learning library that uses **composable compilers** to achieve high performance.

```rust
use luminal::prelude::*;

// Setup graph and tensors
let mut cx = Graph::new();
let a = cx.new_tensor::<R2<3, 1>>("A");
let b = cx.new_tensor::<R2<1, 4>>("B");

// Do stuff...
let c = a.matmul(b);

// Set inputs and mark outputs
a.set(vec![1.0, 2.0, 3.0]);
b.set(vec![1.0, 2.0, 3.0, 3.0]);
c.mark();

// Compile and run graph
cx.compile(GenericCompiler::default());
cx.execute();

// Get result
println!("Result: {:?}", c.data());
```

## Getting Started
Run `cargo run --example simple` to see a super simple example. 

**Llama**
Run `bash examples/llama/setup/setup.sh` and then `cargo run --release --example llama` to start generating text. 

## Why does this look so different from other DL libraries?
Most deep learning libraries are eager-first, meaning each op call directly operates on the data. When you see `x + y`, the addition actually happens right there. This is great for debugging because it works exactly as most developers expect. 

However, this isn't great for performance. What makes sense for a developer doesn't make sense for the machine, in the same way that no one writes assembly by hand. Most libraries try to fix this problem by tacking on operator fusion or JIT compilation to try to change the compilation flow to something better for the machine. Turns out this is [super](https://pytorch.org/docs/stable/dynamo/index.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

If these other libraries are interpreted, luminal is compiled. Luminal takes an approach more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Everything's static here. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph for execution later. Only once `graph.execute()` is ran does the computation happen. *But isn't that just lazy execution?* Yes it is! But in luminal **everything is done this way**. All neural networks are built up as one or a few static computation graphs, and executed later. 

## But Why?
A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, our compilers have global knowledge and can do much more aggressive optimization **without any sync points**.

Of course, we can still split the network into multiple seperate graphs if we want to insert dynamic control flow part-way through, which means this method doesn't preclude optimizations like KV caching, because the KV cached forward pass is just a seperate graph! See `examples/llama` for an example of a KV cache.

Some huge benefits are now unlocked:
- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Devices and Dtypes are handled through compilers (just run the CUDA compiler to convert the graph to use CUDA kernels, then the fp16 compiler to convert to half-precision kernels)
- Networks can be written in generic code, but compiled and ran fast on hyper-specific architectures (try writing a PyTorch network that works with both TF32 dtypes and TPUs; get ready for if statement hell...)

## RISC-style architecture
Luminal can be ran on new accelerators by implementing 12 primitive ops. Take a look at `src/compilers/cuda/prim.rs` to see 1-to-1 CUDA translations of the primops.

Accelerators are free to implement their own custom ops, and their own compilers to convert luminal primitive ops to bespoke ops.

## Compile-time Shape Checks
All operations are shape checked at compile time, so no more shape mismatches! All credit for this goes to [dfdx](https://github.com/coreylowman/dfdx).

## View the Graph
Once you've written all your computation code, run `cx.display()` to see the entire computation graph in all it's glory. Pretty messy looking! Now run `cx.compile(GenericCompiler::default())` and display the graph again. Much better.

## Where are we?
Currently luminal is extremely alpha. Please don't use this in prod.

- Metal and Cuda are supported for running models on Macs and Nvidia GPUs respectively, in both full and half precision.
- Llama 1 is implemented in `examples/llama`. See instructions above for running.
- The llama example shows how to implement a loader for a custom format. Safetensors loaders are already implemented, and are the recommended way to load a model.
- We have a small library of NN modules in `nn`, including transformers.
- A signifigant amount of high-level ops are implemented in `hl_ops`. We are aiming to match the tinygrad ops set.
- Next release will bring a signifigant amount of compilers which should fuse primops into much faster ops. The aim for 0.3 is to be usably fast, not SOTA yet.
- The aim for 0.4 is to achieve SOTA performance on macs, and near SOTA on single nvidia gpus, as well as support all mainstream models (Stable Diffusion, Whisper, Flamingo, etc.)

Some things on the roadmap:
- Optimize cuda and metal matmul kernels
- Build benchmarking suite to test against other libs
- Write specialized Cuda and Metal kernels for full transformer architecture (FlashAttention, etc.)
- Automatic differentiation of graphs
- Beat PT 2.0 perf on LLM training
- Build dyson swarm
