# GraphTensors

We're working with pretty complicated graphs to build our computation on, but we don't want to manually place all the nodes ourselves! So how can we build these static graphs in a nice, familiar way? GraphTensors!

Essentially GraphTensors are pointers to a specific node on the graph, as well as some metadata about the output of that node, such as its shape. We can make a new GraphTensor by doing:
```rust
let mut cx = Graph::new(); // We need a graph to build!
let a: GraphTensor<R1<3>> = cx.tensor(); // Here we create a new node on the graph and get a GraphTensor back, pointing to it.
```
Notice the type of `a`: `GraphTensor<R1<3>>`. So what's that generic all about? It's the shape! We make tensor shapes part of the type, so they're tracked at compile time! In this case, the shape is rank 1, with 3 elements, or in other words, a vector of 3 dimensions. (Side note: `R1<N>` is a typedef of `(Const<N>,)`) It should be impossible to accidentally get a runtime shape mismatch.

Now we can use the `a` as you would in a library like PyTorch, performing linear algebra:
```rust
let b = a.exp().sqrt();
let c = b + a;
```
Looks familiar!

[Let's take a look at how GraphTensors are used to build whole neural networks.](https://github.com/jafioti/luminal/blob/main/docs/03%20Modules.md)