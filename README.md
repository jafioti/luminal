# luminal
![image](https://raw.githubusercontent.com/jafioti/luminal/main/dag.jpeg)
**Deep learning at the speed of light.**

```rust
use luminal::prelude::*;

// Setup graph and tensors
let mut cx = Graph::new();
let a = cx.new_tensor::<R2<3, 1>>();
let b = cx.new_tensor::<R2<1, 4>>();

// Do match
let c = a.matmul(b);

// Set inputs and mark outputs
a.set(vec![1.0, 2.0, 3.0]);
b.set(vec![1.0, 2.0, 3.0, 3.0]);
c.mark();

// Optimize and run graph
cx.optimize(GenericOptimizer::default());
cx.execute();

// Get result
println!("Result: {:?}", c.retrieve().unwrap().data);
```

## Why does this look so different from other DL libraries?
Most deep learning libraries are eager-first, meaning each op call directly operates on the data. So when you see `x + y`, the addition actually happens right there. This is great for debugging, it works exactly as most developers expect. However, this isn't great for performance because what makes sense for a developer doesn't make sense for the machine, in the same way that no one writes assembly by hand. Most libraries try to fix this problem by tacking on operator fusion or JIT compilation to try to change the compilation flow to something better for the machine. Turns out this is [super](https://pytorch.org/docs/stable/dynamo/index.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

Luminal takes a different approach, more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Here everything's static. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph for execution later. Only once `graph.execute()` is ran does the computation happen. *But isn't that just lazy execution?* Yes it is! But in luminal **everything is done this way**. All neural networks are built up as one or a few static computation graphs, and executed later. 

## But Why?
A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, our optimizers have global knowledge and can do much more aggressive optimization **without any sync points**.

Of course, we can still split the network into multiple seperate graphs if we want to insert dynamic control flow part-way through, which means this method doesn't preclude optimizations like KV caching, because the KV cached forward pass is just a seperate graph!

Some huge benefits are now unlocked:
- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Devices and Dtypes are handled through optimizers (just run the CUDA optimizer to convert the graph to use CUDA kernels, then the fp16 optimizer to convert to half-precision kernels)
- Networks can be written in generic code, but compiled and ran fast on hyper-specific architectures (try writing a PyTorch network that works with both TF32 dtypes and TPUs; get ready for if statement hell...)

## RISC-style architecture
Luminal can be implemented on new accelerators by implementing 13 primitive ops.

Accellerators are free to implement their own custom weird ops, and their own optimizers to convert luminal primitive ops to their bespoke ops.

## Compile-time Shape Checks
All operations are shape checked at compile time, so no more shape mismatches! All credit for this goes to [dfdx](https://github.com/coreylowman/dfdx).

## View the Graph
Once you've written all your computation code, run `cx.display_graph()` to see the entire computation graph in all it's glory. Pretty messy looking! Now run `cx.optimize(GeneralOptimizer::default())` and display the graph again. Much better.

## Where are we?
Currently luminal is extremely alpha. Please don't use this in prod.

Some things on the roadmap:
- Write common sense cuda ops and optimizer (matmuls, mul-add, etc.)
- Create reasonable library of NN modules
- Build benchmarking suite to test against other libs
- Write specialized CUDA kernels for full transformer architecture (FlashAttention, etc.)
- Match PT 2.0 perf on LLM training
- Build dyson swarm