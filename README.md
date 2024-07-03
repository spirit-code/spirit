SPIRIT
=============================
**SPIN SIMULATION FRAMEWORK**<br />


![Logo](https://imgur.com/iWc1kuE.png "Spirit Logo")


&nbsp;


**Core Library:**

| Branch   | Build Status | Python Package Coverage | Core Library Coverage |
| :------- | :----------: | :---------------------: | :-------------------: |
| master:  | ![CI](https://github.com/spirit-code/spirit/workflows/CI/badge.svg?branch=master) | [![Coverage Status](https://coveralls.io/repos/github/spirit-code/spirit/badge.svg?branch=master)](https://coveralls.io/github/spirit-code/spirit?branch=master) | [![Coverage Status](https://codecov.io/gh/spirit-code/spirit/branch/master/graph/badge.svg)](https://codecov.io/gh/spirit-code/spirit/branch/master) |
| develop: | ![CI](https://github.com/spirit-code/spirit/workflows/CI/badge.svg?branch=develop) | [![Coverage Status](https://coveralls.io/repos/github/spirit-code/spirit/badge.svg?branch=develop)](https://coveralls.io/github/spirit-code/spirit?branch=develop) | [![Coverage Status](https://codecov.io/gh/spirit-code/spirit/branch/develop/graph/badge.svg)](https://codecov.io/gh/spirit-code/spirit/branch/develop) |

**[Python package](https://pypi.org/project/spirit/):** [![PyPI version](https://badge.fury.io/py/spirit.svg)](https://badge.fury.io/py/spirit)


&nbsp;


The code is released under [MIT License](LICENSE.txt).<br />
If you intend to *present and/or publish* scientific results or visualisations for which you used Spirit,
please cite [`G. P. Müller et al., Phys. Rev. B 99, 224414 (2019)`](https://link.aps.org/doi/10.1103/PhysRevB.99.224414) and read the [docs/REFERENCE.md](docs/REFERENCE.md).

**This is an open project and contributions and collaborations are always welcome!!**
See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) on how to contribute or write an email to moritz.sallermann@rwth-aachen.de or t.puerling@fz-juelich.de<br />
For contributions and affiliations, see [docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md).

Please note that a version of the *Spirit Web interface* is hosted by the Research Centre Jülich at
https://juspin.de


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

**Spirit** is a **platform-independent** framework for spin dynamics, written in C++17.
It combines the traditional cluster work, using the command-line, with modern
visualisation capabilities in order to maximize scientists' productivity.

> "It is unworthy of excellent men to lose hours like slaves in
>  the labour of calculation which could safely be relegated to
>  anyone else if machines were used."
> - Gottfried Wilhelm Leibniz

*Our goal is to build such machines*. The core library of the *Spirit* framework provides an
**easy to use API**, which can be embedded into almost any programming language,
and includes ready-to-use python bindings.
A **powerful desktop user interface** is available, providing real-time visualisation and
control over system parameters.

### *Physics Features*

- Atomistic Spin Lattice [Heisenberg Model](core/docs/Hamiltonian.md) including among others DMI and dipole-dipole
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
- Modular backends including **parallelisation on GPU** (CUDA) and **CPU** (OpenMP and STL parallelisation)

### *Documentation*

More details may be found at [spirit-docs.readthedocs.io](http://spirit-docs.readthedocs.io)
or in the [Reference section](docs/README.md) including

- [Unix/OSX build instructions](docs/Build_Unix_OSX.md)
- [Windows build instructions](docs/Build_Windows.md)
- [Input File Reference](core/docs/Input.md)

There is also a [Wiki](https://iffwiki.fz-juelich.de/index.php/Spirit "Click me..."),
hosted by the Research Centre Jülich.

---------------------------------------------



&nbsp;



Getting started with the Desktop Interface <a name="Desktop"></a>
---------------------------------------------

See the build instructions for [Unix/OSX](docs/Build_Unix_OSX.md) or
[Windows](docs/Build_Windows.md) on how to get the desktop user interface.

![Desktop UI with Isosurfaces in a thin layer](http://imgur.com/QUcN4aG.jpg "Isosurfaces in a thin layer")

The user interface provides a powerful OpenGL visualisation window
using the [VFRendering](https://github.com/FlorianRhiem/VFRendering) library.
It provides functionality to

- Control Calculations
- Locally insert Configurations (homogeneous, skyrmions, spin spiral, …)
- Generate homogeneous Transition Paths
- Change parameters of the Hamiltonian
- Change parameters of the Method and Solver
- Configure the visualisation (arrows, isosurfaces, lighting, …)

See the [UI-QT Reference](docs/UI-Qt.md) for the key bindings of the various features.

*Unfortunately, distribution of binaries for the Desktop UI is not possible due
to the restrictive license on QT-Charts.*

---------------------------------------------



&nbsp;



Getting started with the Python Package <a name="Python"></a>
---------------------------------------------

To install the *Spirit python package*, either build and install from source
([Unix/OSX](docs/Build_Unix_OSX.md), [Windows](docs/Build_Windows.md)) or
simply use

    pip install spirit

With this package you have access to powerful Python APIs to run and control
dynamics simulations or optimizations.
This is especially useful for work on clusters, where you can now script your
workflow, never having to re-compile when testing, debugging or adding features.

The most simple example of a **spin dynamics simulation** would be
``` python
from spirit import state, simulation
with state.State("input/input.cfg") as p_state:
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
```
Where `SOLVER_SIB` denotes the semi-implicit method B and the starting configuration
will be random.

To add some meaningful content, we can change the **initial configuration** by
inserting a Skyrmion into a homogeneous background:
``` python
def skyrmion_on_homogeneous(p_state):
    from spirit import configuration
    configuration.plus_z(p_state)
    configuration.skyrmion(p_state, 5.0, phase=-90.0)
```

If we want to calculate a **minimum energy path** for a transition, we need to generate
a sensible initial guess for the path and use the **GNEB method**. Let us consider
the collapse of a skyrmion to the homogeneous state:
``` python
from spirit import state, chain, configuration, transition, simulation

### Copy the system and set chain length
chain.image_to_clipboard(p_state)
noi = 7
chain.set_length(p_state, noi)

### First image is homogeneous with a Skyrmion in the center
configuration.plus_z(p_state, idx_image=0)
configuration.skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=0)
### Last image is homogeneous
configuration.plus_z(p_state, idx_image=noi-1)
simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=noi-1)

### Create transition of images between first and last
transition.homogeneous(p_state, 0, noi-1)

### GNEB calculation
simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP)
```
where `SOLVER_VP` denotes a direct minimization with the velocity projection algorithm.

You may also use *Spirit* order to **extract quantitative data**, such as the energy.
``` python
def evaluate(p_state):
    from spirit import system, quantities
    M = quantities.get_magnetization(p_state)
    E = system.get_energy(p_state)
    return M, E
```

Obviously you may easily create significantly more complex workflows and use Python
to e.g., pre- or post-process data or to distribute your work on a cluster and much more!
