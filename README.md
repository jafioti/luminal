# luminal
![image](https://raw.githubusercontent.com/jafioti/luminal/main/dag.jpeg)
**Deep learning at the speed of light.**

```rust
let mut cx = Graph::new();
let b = cx.new_tensor::<R2<3, 1>>();
let c = cx.new_tensor::<R2<1, 4>>();

let a = b.matmul(c);

a.mark();
b.set(vec![1.0, 2.0, 3.0]);
c.set(vec![1.0, 2.0, 3.0, 3.0]);

cx.optimize(GeneralOptimizer::default());
cx.execute();

println!("Result: {:?}", a.retrieve().unwrap().data);
```

## Why does this look so different from other DL libraries?
Most deep learning libraries are eager-first, meaning each op call directly operates on the data. So when you see `x + y`, the addition actually happens right there. This is great for debugging as it works exactly as most developers expect it to and is super easy to debug. However, this isn't great for performance, because often times what makes the most sense for a developer doesn't make sense for the machine, and vice versa such as combining many operators into one super-specific kernel no one would think to use. Most libraries try to fix this problem later by tacking on operator fusion or JIT compilation or some other scheme to try to change the compilation flow to something better for the machine. Turns out this is [super](https://pytorch.org/docs/stable/dynamo/index.html) [difficult](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [even](https://pytorch.org/docs/stable/jit.html) [for](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) Pytorch!

Luminal takes a different approach, more similar to [XLA](https://www.tensorflow.org/xla), and [tinygrad](https://github.com/tinygrad/tinygrad). Here everything's static. When you write out an expression like `x + y`, no actual computation happens. The operation is recorded to a directed acyclic computation graph, for execution later. Only once `cx.execute()` is ran does the computation happen. *But isn't that just lazy execution?* Yes it is! But in luminal **everything is done this way**. All neural networks are built up as one or more static computation graphs, and executed later. 

## But Why?
A consequence of this is that the actual computation that gets ran can be radically different than the code that was written. Since we have an entire neural network fully represented in a compute graph, our optimizers have global knowledge and can do much more aggressive optimization **without any sync points**. Of course, we can still split the network into multiple seperate graphs if we want to insert control flow points part-way through, which means this method doesn't preclude optimizations like KV caching, because the KV cached forward pass is just a seperate graph operating on the same data + weights as the normal forward pass graph!

Some huge benefits are now unlocked:
- Aggressive kernel fusion
- Shape-specific kernels compiled at runtime
- Devices are handled through optimizers (just run the CUDA optimizer to convert the graph to use CUDA kernels instead of CPU ones)
- Dtypes are handled through optimizers (just run the f16 optimizer to convert the graph to use half-precision kernels)
- Networks can be written in generic code, but compiled easily to hyper-specific architectures (such as weird accelerators)

## RISC-style architecture
Luminal can be implemented on new accelerators by implementing 13 primitive ops.

Accellerators are free to implement their own custom weird ops, and their own optimizers to convert luminal primitive ops to their bespoke ops.

## Compile-time Shape Checks
All operations are shape checked at compile time, so no more shape mismatches! All credit for this goes to [dfdx](https://github.com/coreylowman/dfdx).

## Where are we?
Currently luminal is extremely alpha. Please don't use this in prod.

Some things on the roadmap:
- Establish comprehensive test suite against PyTorch
- Create NN Module API
- Write CUDA primitive op kernels
- Write CUDA optimizer
- Create reasonable library of NN modules
- Build benchmarking suite to test against other libs
- Write specialized CUDA kernels for full transformer architecture (FlashAttention, etc.)
- Match PT 2.0 perf on LLM training
- Build dyson swarm