Spirit
=============

This is the core library of the **Spirit** framework.

It is meant to provide useful and easy API functions to enable productive work
with Atomistic Dynamics Simulations and Optimizations.
The current implementation is specific to atomistic spin models, but it may
easily be generalised.

The library is written in C++ but has been wrapped in Python for easier install and use.
Other bindings should be easy to create.

### Backends
The core can be parallelized on GPU (CUDA) and CPU (OpenMP). This is realized by
a subset of function definitions being implemented twice, while there is only
one set of function declarations.

The CUDA backend uses unified memory to avoid manual allocations and data transfers
between host and device and to duplicate as little code as possible.

### Further information
* [Input File Reference](docs/Input.md)