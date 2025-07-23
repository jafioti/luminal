# [luminal](https://luminalai.com)
![image](https://github.com/jafioti/luminal/blob/main/docs/images/dag.jpeg)
[![Website](https://img.shields.io/badge/Luminal-Website-blue?style=for-the-badge&color=0D9373)](https://luminalai.com)
[![CI Status](https://img.shields.io/github/actions/workflow/status/jafioti/luminal/test.yml?style=for-the-badge&logo=github-actions&logoColor=white&branch=main)](https://github.com/Sidekick-AI/dataflow/actions)
[![Docs](https://img.shields.io/badge/Documentation-green?style=for-the-badge&color=fc9700)](https://docs.luminalai.com)
[![Current Crates.io Version](https://img.shields.io/crates/v/luminal.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/luminal)
[![discord](https://dcbadge.limes.pink/api/server/PcVsjKHZ)](https://discord.gg/AfGCbHaX)

**Deep learning at the speed of light.**

Luminal is a deep learning library that **generates optimized code** to achieve high performance.

> [!IMPORTANT]  
> Recently significant work is happening to focus much more on a search-first compiler with the goal of discovering advanced kernels like FlashAttention automatically. See progress on the `next` branch.

```rust
use luminal::prelude::*;

// Setup graph and tensors
let mut cx = Graph::new();
let a = cx.tensor((3, 1)).set([[1.0], [2.0], [3.0]]);
let b = cx.tensor((1, 4)).set([[1.0, 2.0, 3.0, 4.0]]);

// Do math...
let mut c = a.matmul(b).retrieve();

// Compile and run graph
cx.compile(<(GenericCompiler, CPUCompiler)>::default(), &mut c);
cx.execute();

// Get result
println!("Result: {:?}", c);
```

## Getting Started
**Llama 3 8B**
- the below is a quick example of how you can run Llama 3 8B locally using Luminal
    - to go indepth on this example check out [the documentation here](https://github.com/jafioti/luminal/tree/main/examples/llama/README.md)
```bash
cd ./examples/llama
# Download the model
bash ./setup/setup.sh
# Run the model
cargo run --release --features metal    # MacOS (Recommended)
cargo run --release --features cuda     # Nvidia
cargo run --release                     # CPU
```

## Features
### Speed
Luminal can run Q8 Llama 3 8B on M-series Macbooks at 15-25 tokens per second. The goal is to become the fastest ML framework for any model on any device.

### Simplicity
The core of luminal is and always will be minimal. It should be possible to understand the entire core library in an afternoon.

### RISC-style architecture
Everything in luminal boils down to 12 primitive ops:
- Unary - `Log2, Exp2, Sin, Sqrt, Recip`
- Binary - `Add, Mul, Mod, LessThan`
- Other - `SumReduce, MaxReduce, Contiguous`

These ops are enough to support transformers, convnets, etc.

### Speed
We compile these ops into complex GPU kernels, so even though our ops are simple, we get high performance through the power of compilers! This is how we overcome the typical RISC disadvantages, btw. 

### Native
The current ML ecosystem is too fragmented, and the solution isn't another layer of abstraction. Luminal is written in rust, and interacts directly with the CUDA / Metal APIs. No indirections or abstractions, docker containers, or virtual environments. Just a statically-linked rust crate.

### Validated against Pytorch
Correctness matters. So we write as much tests as possible to cover all ops and verify they work the same as an equivalent Pytorch implementation. ([Improvements needed!](https://github.com/jafioti/luminal/issues/20))

## Ideology
### Why does this look so different from other DL libraries?
Most deep learning libraries are eager-first, meaning each op call directly operates on the data. In PyTorch, when you see `x + y`, the addition actually happens right there. This is great for debugging because it works exactly as most developers expect.

However, this isn't great for performance. What makes sense for a developer doesn't work well for the machine, in the same way that no one writes assembly by hand. Most libraries try to fix this problem by tacking on operator fusion or JIT compilation to try to change the compilation flow to something better for the machine. Turns out this is [super](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

### Compile everything
A core tenet of Luminal is ahead-of-time compilation. Whenever possible, push everything to compile time and leave nothing to run time. Luminal takes an approach more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Everything's static here. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph for execution later. Only once `graph.execute()` is ran does the computation happen. *But isn't that just lazy execution?* Yes it is! But in luminal **everything is done this way**. All neural networks are built up as one or a few static computation graphs, compiled, and executed later.

**But why?**

A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, our compilers have global knowledge. This means we can push most ML complexity to the compilers. For instance, devices, datatypes, and execution schedules are all handled by compliers. Even autograd is handled by a compiler!

Now we can do:
- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Devices and Dtypes are handled through compilers (just run the CUDA compiler to convert the graph to use CUDA kernels, then the fp16 compiler to convert to half-precision kernels)
- Networks can be written in generic code, but compiled and ran fast on hyper-specific architectures (try writing a PyTorch network that works with both TF32 dtypes and TPUs; get ready for if statement hell...)

### View the Graph
Once you've written all your computation code, run `cx.display()` to see the entire computation graph in all it's glory. Pretty messy looking! Now run `cx.compile(GenericCompiler::default())` and display the graph again. Much better.

## Where are we?
- Metal and Cuda are supported for running models on Macs and Nvidia GPUs respectively, in both full and half precision.
- Performance on M-series macs with LLMs is within 20% of llama.cpp (a *heavily* optimized library)
- Full training support with graph-based autograd.
- Llama 3, Phi 3, Whisper and Yolo v8 are implemented in `examples/`. See instructions above for running.
- We have a small library of NN modules in `luminal_nn`, including transformers.
- A significant amount of high-level ops are implemented in `hl_ops`. We are aiming to match the most used ~80% of the pytorch api.
- The aim for 0.3 is to achieve SOTA performance on an M1 pro (50 tok/s), and near SOTA on single nvidia gpus (>100 tok/s), as well as support many mainstream models (Whisper, Stable Diffusion, Yolo v9, etc.) See the tracking issue [here](https://github.com/jafioti/luminal/issues/29)

Some things on the roadmap:
- Optimize cuda and metal matmul kernels
- Fine-grained metal and cuda IR
- Build benchmarking suite to test against other libs
- Distributed data, pipeline and tensor parallel.
- Beat PT 2.0 perf on LLM training
- Write compiler for quantum photonic retro encabulator
- Build dyson swarm

## License
Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
