SPIRIT
=============================
**SPIN SIMULATION FRAMEWORK**<br />


![Logo](https://imgur.com/iWc1kuE.png "Spirit Logo")


&nbsp;


**Core Library:**

| Service | System | Compiler | Status |
| :------ | :----: | :------: | -----: |
|  [Travis-CI](https://travis-ci.org/spirit-code/spirit) | Ubuntu 14.04 <br/> macOS | GCC 6 <br/> Clang | master: [![Build Status](https://travis-ci.org/spirit-code/spirit.svg?branch=master)](https://travis-ci.org/spirit-code/spirit)<br />develop: [![Build Status](https://travis-ci.org/spirit-code/spirit.svg?branch=develop)](https://travis-ci.org/spirit-code/spirit) |
|  [AppVeyor](https://ci.appveyor.com/project/GPMueller/spirit) | Windows | MSVC14 <br/> MSVC14.1 | master: [![Build status](https://ci.appveyor.com/api/projects/status/m15hlxkgmuyx8aj5/branch/master?svg=true)](https://ci.appveyor.com/project/GPMueller/spirit)<br />develop: [![Build status](https://ci.appveyor.com/api/projects/status/m15hlxkgmuyx8aj5/branch/develop?svg=true)](https://ci.appveyor.com/project/GPMueller/spirit) |


**[Python package](https://pypi.org/project/spirit/):** [![PyPI version](https://badge.fury.io/py/spirit.svg)](https://badge.fury.io/py/spirit)

| Branch   | Python Package Coverage | Core Library Coverage |
| :------- | :---------------------: | :-------------------: |
| master:  | [![Coverage Status](https://coveralls.io/repos/github/spirit-code/spirit/badge.svg?branch=master)](https://coveralls.io/github/spirit-code/spirit?branch=master) | [![Coverage Status](https://codecov.io/gh/spirit-code/spirit/branch/master/graph/badge.svg)](https://codecov.io/gh/spirit-code/spirit/branch/master) |
| develop: | [![Coverage Status](https://coveralls.io/repos/github/spirit-code/spirit/badge.svg?branch=develop)](https://coveralls.io/github/spirit-code/spirit?branch=develop) | [![Coverage Status](https://codecov.io/gh/spirit-code/spirit/branch/develop/graph/badge.svg)](https://codecov.io/gh/spirit-code/spirit/branch/develop) |


&nbsp;


The code is released under [MIT License](LICENSE.txt).<br />
If you intend to *present and/or publish* scientific results or visualisations for which you used Spirit,
please cite `G. P. Müller et al., arXiv:1901.11350` and read the [REFERENCE.md](docs/REFERENCE.md).

**This is an open project and contributions and collaborations are always welcome!!**
See [CONTRIBUTING.md](docs/CONTRIBUTING.md) on how to contribute or write an email to g.mueller@fz-juelich.de<br />
For contributions and affiliations, see [CONTRIBUTORS.md](docs/CONTRIBUTORS.md).

Please note that a version of the *Spirit Web interface* is hosted by the Research Centre Jülich at
http://juspin.de


&nbsp;

<!--
![nur ein Beispiel](https://commons.wikimedia.org/wiki/File:Example_de.jpg "Beispielbild")
-->

![Skyrmions](http://imgur.com/JgPj8t5.jpg "Skyrmions on a 2D grid")

&nbsp;



Contents
--------

1. [Introduction](#Introduction)
2. [Getting started with the Desktop User Interface](#Desktop)
3. [Getting started with the Python Package](#Python)

---------------------------------------------



&nbsp;



Introduction <a name="Introduction"></a>
---------------------------------------------

#### A modern framework for magnetism science on clusters, desktops & laptops and even your Phone

**Spirit** is a **platform-independent** framework for spin dynamics, written in C++11.
It combines the traditional cluster work, using using the command-line, with modern
visualisation capabilites in order to maximize scientists' productivity.

> "It is unworthy of excellent men to lose hours like slaves in
>  the labour of calculation which could safely be relegated to
>  anyone else if machines were used."
> - Gottfried Wilhelm Leibniz

*Our goal is to build such machines*. The core library of the *Spirit* framework provides an
**easy to use API**, which can be used from almost any programming language,
and includes ready-to-use python bindings.
A **powerful desktop user interface** is available, providing real-time visualisation and
control of parameters.

### *Physics Features*

- Atomistic Spin Lattice Heisenberg Model including also DMI and dipole-dipole
- **Spin Dynamics simulations** obeying the
  [Landau-Lifschitz-Gilbert equation](https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation)
- Direct **Energy minimisation** with different solvers
- **Minimum Energy Path calculations** for transitions between different
  spin configurations, using the GNEB method

### *Highlights of the Framework*

- Cross-platform: everything can be built and run on Linux, OSX and Windows
- Standalone core library with C API which can be used from almost any programming language
- **Python package** making complex simulation workflows easy
- Desktop UI with powerful, live **3D visualisations** and direct control of most system parameters
- Modular backends including **parallelisation on GPU** (CUDA) and **CPU** (OpenMP) 

### *Documentation*

More details may be found at [spirit-docs.readthedocs.io](http://spirit-docs.readthedocs.io)
or in the [Reference section](docs/README.md) including

- [Framework build instructions](docs/BUILD.md)
- [Core build instructions](core/docs/BUILD.md)
- [Core API Reference](core/docs/API.md)
- [Input File Reference](core/docs/INPUT.md)

There is also a [Wiki](https://iffwiki.fz-juelich.de/index.php/Spirit "Click me..."),
hosted by the Research Centre Jülich.

---------------------------------------------



&nbsp;



Getting started with the Desktop Interface <a name="Desktop"></a>
---------------------------------------------

See [BUILD.md](docs/BUILD.md) on how to install the desktop user interface.

![Desktop UI with Isosurfaces in a thin layer](http://imgur.com/QUcN4aG.jpg "Isosurfaces in a thin layer")

The user interface provides a powerful OpenGL visualisation window
using the [VFRendering](https://github.com/FlorianRhiem/VFRendering) library.
It provides functionality to

- Control Calculations
- Locally insert Configurations (homogeneous, skyrmions, spin spiral, ... )
- Generate homogeneous Transition Paths
- Change parameters of the Hamiltonian
- Change parameters of the Method and Solver
- Configure the Visualization (arrows, isosurfaces, lighting, ...)

See the [UI-QT Reference](docs/UI-Qt.md) for the key bindings of the various features.

*Unfortunately, distribution of binaries for the Desktop UI is not possible due
to the restrictive license on QT-Charts.*

---------------------------------------------



&nbsp;

 

Getting started with the Python Package <a name="Python"></a>
---------------------------------------------

To install the *Spirit python package*, either [build and install from source](docs/BUILD.md)
or simply use

    pip install spirit

With this package you have access to powerful Python APIs to run and control
dynamics simulations or optimizations.
This is especially useful for work on clusters, where you can now script your
workflow, never having to re-compile when testing, debugging or adding features.

The most simple example of a **spin dynamics simulation** would be
``` python
    from spirit import state, simulation
    with state.State("input/input.cfg") as p_state:
        simulation.PlayPause(p_state, "LLG", "SIB")
```
Where `"SIB"` denotes the semi-implicit method B and the starting configuration
will be random.

To add some meaningful content, we can change the **initial configuration** by
inserting a Skyrmion into a homogeneous background:
``` python
    def skyrmion_on_homogeneous(p_state):
        from spirit import configuration
        configuration.PlusZ(p_state)
        configuration.Skyrmion(p_state, 5.0, phase=-90.0)
```

If we want to calculate a **minimum energy path** for a transition, we need to generate
a sensible initial guess for the path and use the **GNEB method**. Let us consider
the collapse of a skyrmion to the homogeneous state:
``` python
    from spirit import state, chain, configuration, transition, simulation 

    ### Copy the system a few times
    chain.Image_to_Clipboard(p_state)
    for number in range(1,7):
        chain.Insert_Image_After(p_state)
    noi = chain.Get_NOI(p_state)

    ### First image is homogeneous with a Skyrmion in the center
    configuration.PlusZ(p_state, idx_image=0)
    configuration.Skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
    simulation.PlayPause(p_state, "LLG", "VP", idx_image=0)
    ### Last image is homogeneous
    configuration.PlusZ(p_state, idx_image=noi-1)
    simulation.PlayPause(p_state, "LLG", "VP", idx_image=noi-1)

    ### Create transition of images between first and last
    transition.Homogeneous(p_state, 0, noi-1)

    ### GNEB calculation
    simulation.PlayPause(p_state, "GNEB", "VP")
```
where `"VP"` denotes a direct minimization with the velocity projection algorithm.

You may also use *Spirit* order to **extract quantitative data**, such as the energy.
``` python
    def evaluate(p_state):
        from spirit import system, quantities
        M = quantities.Get_Magnetization(p_state)
        E = system.Get_Energy(p_state)
        return M, E
```

Obviously you may easily create significantly more complex workflows and use Python
to e.g. pre- or post-process data or to distribute your work on a cluster and much more!