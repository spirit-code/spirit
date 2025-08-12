Custom Interaction (Tutorial)
====================================================

Implementing a custom interaction is straightforward in Spirit.

The standard interaction in Spirit has a notion of locality.
That means that you only have to define locally how energy, gradient and Hessian are calculated, the Hamiltonian handles the remaining parts -- especially parallelization -- for you.
If you need even more customization the last section explains how to implement a non-local interaction.


Setup
----------------------------------------------------

First, clone the spirit repository and follow the instructions so that you can build spirit locally:
- [Unix/OSX](/docs/Build_Unix_OSX.md#core-library)
- [Windows](/docs/Build_Windows.md#core-library)

Then install the developer version of the spirit python package:
```sh
pip install --editable "core/python[dev]" --user
```

This provides you with the `spirit-mkinteraction` CLI utility which we are going to use to generate the boilerplate code necessary for the interaction.
Run `spirit-mkinteraction --help` for details on how to use it.

The tool will generate an empty interaction with the chosen name for you.
In case you want to look at an instructive example you can also use the `--demo` option to generate a custom version of the two-site anisotropy interaction to play with.


Local Interaction
----------------------------------------------------

The core of each interaction is how it computes the energy density, local gradient and Hessian and which data it needs to do that.
Added to that are some utility functions that the Hamiltonian expects to get brief metrics on the interaction.

If you generate a local interaction you will be left with two files to edit: the implementation file in `core/include/engine/spin/interaction/` and the IO file in `core/src/io/hamiltonian/`.
Both of these are commented to give an understanding of what each component does.
The following sections serve as a supplementary explanation of the concepts behind the code.

### Local Functors

The Hamiltonian expects to get the local energy, gradient and Hessian from callable objects, which in C++ are usually called functors.
These are defined in the struct representing the interaction as nested types.
In principle, they can be customized as long as they adhere to the constructor and call signature described in the generated file, but typically the provided templates that only require you to override the call operator are sufficient.
Each of them get as inputs a `Span` of `Index` objects and a reference to the state object containing all spin orientations and are expected to return.

The `Energy` functor is expected to return a `scalar` representing the local energy density at spin site `i`.
It is convention to store the index that the functor currently calculates in the `ispin` member variable of each `Index` object.
The `Gradient` functor works the same, but should return a 3-vector (`Vector3`) instead.
The `Hessian` gets a callable that is capable of setting an entry for an index pair.

The last important functor is the `Energy_Single_Spin` functor, which is used to calculate the energy difference for two configurations that only differ in a single site.
The provided template scales the energy density at that site by the factor provided as the second template argument.
Its call signature is the same as that of the `Energy` functor.


### Local Datatypes

The constructor and internal data of the functors is inherited from the `DataRef` struct.
It has to store references to the `Interaction`, `Cache` and `Data` types associated with the interaction as well as the public `is_contributing` member variable.
It is also used to store any references to the actual data that is required in each functor.
Most aspects can be customized, but required are the signature of the constructor; the three nested type aliases for `Interaction`, `Data` and `Cache`; and the public `is_contributing` member variable.
It is also advisable to reference any data collections as bare pointers or using a `Span`, otherwise the CUDA build won't work.

`Index` is a unique type defined inside the interaction struct.
It contains indices to all values that are necessary to compute a local energy, gradient or Hessian.
As such it can be thought of representing a spin-cluster together with pointers to the required tensors.

The `Data` and `Cache` struct store the configuration of the interaction.
The `Data` struct is set directly from the configuration file while the `Cache` struct can be updated based on the `Geometry`.
While its use is optional it has to be defined at least as an empty struct.


Non-local Interaction
----------------------------------------------------


Using the local interaction template is advisable for most custom interactions.
If you do however require more customization we also provide a non-local version of the interaction.
This leaves you in charge of any parallelization that in local interactions is handled for you.
The demo version provided by `spirit-mkinteraction --demo` generates a non-local version of the two-site interaction.

After generating a non-local interaction you will be left with three files to edit: two implementation files in `core/include/engine/spin/interaction/` and `core/src/engine/spin/interaction/` respectively, as well as the IO file in `core/src/io/hamiltonian/`.

### Non-local Functors

The contract for non-local interactions is weaker, but requires more implementation from the user.
The main difference is that it doesn't define a `Index` struct so it is more customizable in which interactions it can represent.
The functors now operate on the full density, which they are expected to add their contribution to.


### Non-local Datatypes


The `Data` and `Cache` types for this example are the same as in the [local section](#local-datatypes)

What is different is the `DataRef` struct.
Since it is not expected to be passed directly to any CUDA kernel it stores references to the full `Data` and `Cache` objects within the Hamiltonian.
The reference to the `Cache` object is mutable while the reference to the `Data` object is constant.


Configparser
----------------------------------------------------


This file defines how data is exchanged with the configuration file.
Spirit uses [`toml++`](https://marzer.github.io/tomlplusplus/) to convert from and to TOML.
In addition to that we provide a `TableParser` utility that is designed to parse table-like data with heterogeneous data types in arbitrary order.


### Tableparser

There are two kinds of signatures to define the data types of the columns.
The `TableParser` template takes a bare sequence of types which is useful for small tables.
Larger tables should be initialized using the `TableParserInit` alias template.
It parses the types passed to it as `<type, count>`-pairs packaged as `std::array` types.
The column labels are passed to the constructor and have to match the number of provided types.
Since the parser lowercases any column label it encounters in the configuration file, the passed in labels also have to be lowercase.

The `TableParser::parse()` method expects a `Filter_File_Handle` as the first argument.
An optional callable (2nd order function) can be passed to the `transform_factory` parameter to process the incoming data while it gets parsed.
This is mainly useful, when different configurations of columns would be valid.
