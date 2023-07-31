## Luminal Introduction

Let's get up to speed with how to use luminal, and how it works internally.

First we'll take a look at what the simplest program will look like:
```rust
use luminal::prelude::*;

// Setup graph and tensors (1)
let mut cx = Graph::new();
let a = cx.new_tensor::<R1<3>>();
let b = cx.new_tensor::<R1<3>>();

// Actual operations (2)
let c = a + b;

// Set inputs and mark outputs (3)
a.set(vec![1.0, 2.0, 3.0]);
b.set(vec![1.0, 2.0, 3.0]);
c.mark();

// Run graph (4)
cx.execute();

// Get result (5)
println!("Result: {:?}", c.retrieve().unwrap().real_data(c.view().unwrap()).unwrap());
// Prints out [2.0, 4.0, 6.0]
```
Wow! A lot is going on here just to add two tensors together. That's because luminal isn't really designed for such simple computation, and there's little benifit to using it here. But we'll see it pay off when we start doing more complex operations.

So what's happening here?
1) We're setting up a new `Graph` which tracks all computation and actually does execution. We're also defining two new tensors, both of shape (3,). At this point, these "tensors" are actually `GraphTensor`s that don't hold any data. Also, notice we pass in the shape as a type generic. *Types are known at compile time, similar to [dfdx](https://github.com/coreylowman/dfdx)!*
2) Now we can start doing the thing we came here for: the addition. So we add two `GraphTensor`s together, and get a new `GraphTensor`. Notice this *does not* consume anything, and we're free to use a or b later on. This is because `GraphTensor` is a super lightweight tracking struct which implements copy. "But wait, we never set tbe values of a and b, how can we add them? **We aren't actually adding them here.** Instead, we're writing this addition to the graph, and getting out c, which points to the result when it's actually done.
3) Then we set the data for these tensors. But if `GraphTensor` doesn't hold data, how can we set it? Well we aren't actually setting it *in* the tensor, just passing it through to the graph to say *once you run, set this tensor to this value.* We also need to mark the output we want to retrieve later. This is so that when the graph runs, it doesn't delete the data for c part-way through execution (a common optimization for unused tensors). Notice we're setting the sources *after* we define the computation. This is backward from a lot of other libs, but it means we can redefine the data and rerun everything without redefining the computation later on.
4) Once we call `cx.execute()`, we've already set all our sources, so our addition actually gets ran and stored in c!
5) Now since we're done computing c, we can fetch the data for c and see the result. *This API is likely to change, as it's very ugly.*

Alright, that was a lot but now we've touched on all the main aspects of running a model in luminal.

## NN Modules
Like any good DL library, we organize our networks into `Module`s. Here is the module trait:
```rust
/// A module with a forward pass
pub trait Module<I> {
    type Output;
    fn forward(&self, input: I) -> Self::Output;
}
```
Super simple, we just define a forward function that takes an input and returns an output. A consequence of this is it allows us to define seperate forward passes for single and batched inputs!

Now let's take a look at how `Linear` is defined:
```rust
/// A simple linear layer
pub struct Linear<const A: usize, const B: usize> {
    pub(crate) weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> Module<GraphTensor<R1<A>>> for Linear<A, B> {
    type Output = GraphTensor<R1<B>>;

    fn forward(&self, input: GraphTensor<R1<A>>) -> Self::Output {
        input.matmul(self.weight)
    }
}
```
Here we see a single weight matrix as the internal state, of size AxB. We've written a single forward function for single input vectors of shape (A,) and matmul it by our weight matrix to get an output of shape (B,).

Again, notice we're only dealing with `GraphTensor`s here, so when this code actually gets ran, **no computation happens, it just gets recorded to the graph.**
