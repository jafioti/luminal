# Contributing to luminal
![image](https://raw.githubusercontent.com/jafioti/luminal/main/resources/dag.jpeg)

Please take a look at the [issues](https://github.com/jafioti/luminal/issues) and [roadmap](https://github.com/users/jafioti/projects/1) to see what's targeted for upcoming releases. Contributions for those features are preferred and will be reviewed and merged very rapidly. Other contributions are welcome, but please note luminal is and always will be a fairly minimal library.

The core design of luminal is heavily predicated on extensibility. Compilers alow for immense complexity to be removed from the core library and added with third party compilers. For instance, datatypes and devices are typically first class primitives. In luminal, they're compilers and the core has no idea about them. This is the general trend we'll stick to: core remains brutally simple, and everything that can be externalized to a compiler will be.

We will be adding training support soon, and as you guessed, it will entirely reside in a compiler. Just define the model's graph, run the output through an optimizer, and then run the `AutogradCompiler` before any other compilers. Boom, we got training, and the core of the library has no idea! (aside from some quality of life apis)

PRs that remove complexity are always welcome, but note that line count often is a bad proxy for complexity. Ideally the entire luminal core should be a few thousand lines of code, but anything remotely resembling code golf is not allowed.
