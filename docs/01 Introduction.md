# Luminal Introduction

Let's get up to speed with how to use luminal, and how it works internally.

First we'll take a look at what the simplest program will look like:
```rust
use luminal::prelude::*;

// Setup graph and tensors (1)
let mut cx = Graph::new();
let a = cx.new_tensor::<R1<3>>()
    .set(vec![1.0, 2.0, 3.0]);
let b = cx.new_tensor::<R1<3>>()
    .set(vec![1.0, 2.0, 3.0]);

// Actual operations (2)
let c = (a + b).retrieve();

// Run graph (3)
cx.execute();

// Get result (4)
println!("Result: {:?}", c);
// Prints out [2.0, 4.0, 6.0]
```
Wow! A lot is going on here just to add two tensors together. That's because luminal isn't really designed for such simple computation, and there's little benifit to using it here. But we'll see it pay off when we start doing more complex operations.

So what's happening here?
1) We're setting up a new `Graph` which tracks all computation and actually does execution. We're also defining two new tensors, both of shape (3,). At this point, these "tensors" are actually `GraphTensor`s that don't hold any data. Also, notice we pass in the shape as a type generic. *Types are known at compile time, similar to [dfdx](https://github.com/coreylowman/dfdx)!*
2) Now we can start doing the thing we came here for: the addition. So we add two `GraphTensor`s together, and get a new `GraphTensor`. Notice this *does not* consume anything, and we're free to use a or b later on. This is because `GraphTensor` is a super lightweight tracking struct which implements copy. "But wait, we never set tbe values of a and b, how can we add them? **We aren't actually adding them here.** Instead, we're writing this addition to the graph, and getting out c, which points to the result when it's actually done.

Then we set the data for these tensors. But if `GraphTensor` doesn't hold data, how can we set it? Well we aren't actually setting it *in* the tensor, just passing it through to the graph to say *once you run, set this tensor to this value.* We also need to mark the output we want to retrieve later. This is so that when the graph runs, it doesn't delete the data for c part-way through execution (a common optimization for unused tensors). Notice we're setting the sources *after* we define the computation. This is backward from a lot of other libs, but it means we can redefine the data and rerun everything without redefining the computation later on.
3) Once we call `cx.execute()`, we've already set all our sources, so our addition actually gets ran and stored in c!
4) Now since we're done computing c, we can fetch the data for c and see the result.

Alright, that was a lot but now we've touched on all the main aspects of running a model in luminal.

[Let's take a look at each piece in more depth.](https://github.com/jafioti/luminal/blob/main/docs/02%20GraphTensor%20API.md)