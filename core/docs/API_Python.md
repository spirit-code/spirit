SPIRIT Python API
====================

State
-----

To create a new state with one chain containing a single image, initialized by an [input file](INPUT.md), and run the most simple example of a **spin dynamics simulation**:
```python
from spirit import state
from spirit import simulation

cfgfile = "input/input.cfg"                     # Input File
with state.State(cfgfile) as p_state:           # State setup
    simulation.PlayPause(p_state, "LLG", "SIB") # Start a LLG simulation using the SIB solver
```
or call setup and delete manually:
```python
from spirit import state
from spirit import simulation

cfgfile = "input/input.cfg"                 # Input File
p_state = state.setup(cfgfile)              # State setup
simulation.PlayPause(p_state, "LLG", "SIB") # Start a LLG simulation using the SIB solver
state.delete(p_state)                       # State cleanup
```

You can pass a [config file](INPUT.md) specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
**Note that you currently cannot change the geometry of the systems in your state once they are initialized.**

| State manipulation                                                                  | Returns    |
| ----------------------------------------------------------------------------------- | ---------- |
| `setup( configfile="", quiet=False )`                                               | `None`     |
| `delete(p_state )`                                                                  | `None`     |


Chain
-----


For having more images one can copy the active image in the Clipboard and then insert in a specified position of the chain.
```python
chain.Image_to_Clipboard(p_state )   # Copy p_state to Clipboard
chain.Insert_Image_After(p_state )   # Insert the image from Clipboard right after the currently active image
```
For getting the total number of images in the chain
```python
number_of_images = chain.Get_NOI(p_state )
```

| Get Info                                                                    | Returns        | Description                                                |
| --------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `Get_Index(p_state )`                                                       | `int`          | Get Chain index                                            |
| `Get_NOI(p_state, idx_chain=-1)`                                            | `int`          | Get Chain number of images                                 |
| `Get_Rx(p_state, idx_chain=-1)`                                             | `Array`        | Get Rx                                                     |
| `Get_Rx_Interpolated(p_state, idx_chain=-1)`                                | `Array(float)` | Get Rx interpolated                                        |
| `Get_Energy(p_state, idx_chain=-1)`                                         | `Array(float)` | Get Energy of every System in Chain                        |
| `Get_Energy_Interpolated(p_state, idx_chain=-1)`                            | `Array(float)` | Get interpolated Energy of every System in Chain           |

| Image Manipulation                                              | Returns        | Description                                                |
| --------------------------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `Next_Image(p_state, idx_chain=-1)`                             | `None`         | Switch active to next image of chain (one with largest index). If the current active is the last there is no effect. |
| `Prev_Image(p_state, idx_chain=-1)`                             | `None`         | Switch active to previous image of chain (one with smaller index). If the current active is the first one there is no effect |
| `Jump_To_Image(p_state, idx_image=-1, idx_chain=-1)`            | `None`         | Switch active to specific image of chain. If this image does not exist there is no effect.          |
| `Image_to_Clipboard(p_state, idx_image=-1, idx_chain=-1)`       | `None`         | Copy active image to clipboard                             |
| `Replace_Image(p_state, idx_image=-1, idx_chain=-1)`            | `None`         | Replace active image in chain. If the image does not exist there is no effect.                      |
| `Insert_Image_Before(p_state, idx_image=-1, idx_chain=-1)`      | `None`         | Inserts clipboard image before the current active image. Active image index is increment by one.    |
| `Insert_Image_After(p_state, idx_image=-1, idx_chain=-1)`       | `None`         | Insert clipboard image after the current active image. Active image has the same index.             |
| `Push_Back(p_state, idx_chain=-1)`                              | `None`         | Insert clipboard image at end of chain (after the image with the largest index).                    |
| `Delete_Image(p_state, idx_image=-1, idx_chain=-1)`             | `None`         | Delete active image. If index is specified delete the corresponding image. If the image does not exist there is no effect. |
| `Pop_Back(p_state, idx_chain=-1)`                               | `None`         | Delete image at end of chain.    |

| Data                                        | Returns        | Description                                                |
| ------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `Update_Data(p_state, idx_chain=-1)`        | `None`         | Update the chain's data (interpolated energies etc.)       |
| `Setup_Data(p_state, idx_chain=-1)`         | `None`         | Setup the chain's data arrays                              |


System
------

| System                                                                | Returns  | Description                                                                          |
| --------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------ |
| `Get_Index(p_state)`                                                  | `int`    | Returns the index of the currently active image                                      |
| `Get_NOS(p_state, idx_image=-1, idx_chain=-1)`                        | `int`    | Returns the number of spins                                                          |
| `Get_Spin_Directions(p_state, idx_image=-1, idx_chain=-1)`            | `[3*NOS]`| Returns an `numpy.Array` of size `3*NOS` with the components of each spin's vector   |
| `Get_Energy(p_state, idx_image=-1, idx_chain=-1)`                     | `float`  | Returns the energy of the system                                                     |
| `Update_Data(p_state, idx_image=-1, idx_chain=-1)`                    | `None`   | Update the data of the state                                                         |
| `Print_Energy_Array(p_state, idx_image=-1, idx_chain=-1)`             | `None`   | Print the energy array of the state                                                  |


Constants
---------

| Physical Constants                      | Returns        | Description                                        |
| --------------------------------------- | -------------- | -------------------------------------------------- |
| `mu_B()`                                | `float`        | The Bohr Magneton [meV / T]                        |
| `k_B()`                                 | `float`        | The Boltzmann constant [meV / K]                   |
| `hbar()`                                | `float`        | Planck's constant over 2pi [meV*ps / rad]          |
| `mRy()`                                 | `float`        | Millirydberg [mRy / meV]                           |
| `gamma()`                               | `float`        | The Gyromagnetic ratio of electron [rad / (ps*T)]  |
| `g_e()`                                 | `float`        | The Electron g-factor [unitless]                   |

Geometry
--------

| Get Geometry parameters                                              | Returns               | Description                                        |
| -------------------------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `Get_Bounds(p_state, idx_image=-1, idx_chain=-1)`                    | `[3], [3]`            | Get bounds (minimum and maximum arrays)            |
| `Get_Center(p_state, idx_image=-1, idx_chain=-1)`                    | `float, float, float` | Get center                                         |
| `Get_Basis_Vectors(p_state, idx_image=-1, idx_chain=-1)`             | `[3],[3],[3]`         | Get basis vectors                                  |
| `Get_N_Cells(p_state, idx_image=-1, idx_chain=-1)`                   | `Int, Int, Int`       | Get number of  cells in each dimension             |
| `Get_Translation_Vectors(p_state, idx_image=-1, idx_chain=-1)`       | `[3],[3],[3]`         | Get translation vectors                            |
| `Get_Dimensionality(p_state, idx_image=-1, idx_chain=-1)`            | `int`                 | Get dimensionality of the system                   |
| `Get_Spin_Positions(p_state, idx_image=-1, idx_chain=-1)`            | `[3*NOS]`             | Get Spin positions                                 |
| `Get_Atom_Types(p_state, idx_image=-1, idx_chain=-1)`                | `[NOS]`               | Get atom types                                     |

Hamiltonian
-----------

| Set Parameters                                                                  | Returns               | Description                                        |
| ------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `Set_Field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)`          | `None`                | Set external magnetic field                        |
| `Set_Anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)`     | `None`                | Set anisotropy                                     |


Log
---

| Log manipulation                                                         | Returns   | Description                 |
| ------------------------------------------------------------------------ | --------- | --------------------------- |
| `Send(p_state, level, sender, message, idx_image=-1, idx_chain=-1)`      | `None`    | Send a Log message          |
| `Append(p_state)`                                                        | `None`    | Append Log to file          |


Parameters
----------

### LLG

| Set LLG Parameters                                                                            | Returns       |
| --------------------------------------------------------------------------------------------- | ------------- |
| `setIterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`          | `None`        |
| `setDirectMinimization(p_state, use_minimization, idx_image=-1, idx_chain=-1)`                | `None`        |
| `setConvergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                            | `None`        |
| `setTimeStep(p_state, dt, idx_image=-1, idx_chain=-1)`                                        | `None`        |
| `setDamping(p_state, damping, idx_image=-1, idx_chain=-1)`                                    | `None`        |
| `setSTT(p_state, use_gradient, magnitude, direction, idx_image=-1, idx_chain=-1)`             | `None`        |
| `setTemperature(p_state, temperature, idx_image=-1, idx_chain=-1)`                            | `None`        |

| Get LLG Parameters                                                     | Returns       |
| ---------------------------------------------------------------------- | ------------- |
| `getIterations(p_state, idx_image=-1, idx_chain=-1)`                   | `int, int`    |
| `getDirect_Minimization(p_state, idx_image=-1, idx_chain=-1)`          | `int`         |
| `getConvergence(p_state, idx_image=-1, idx_chain=-1)`                  | `float`       |
| `getTimeStep(p_state, idx_image=-1, idx_chain=-1)`                     | `float`       |
| `getDamping(p_state, idx_image=-1, idx_chain=-1)`                      | `float`       |
| `getSTT(p_state, idx_image=-1, idx_chain=-1)`                          | `float, [3], bool` |
| `getTemperature(p_state, idx_image=-1, idx_chain=-1)`                  | `float`       |

### GNEB

| Set GNEB Parameters                                                                                | Returns       |
| -------------------------------------------------------------------------------------------------- | ------------- |
| `setIterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`               | `None`        |
| `setConvergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                                 | `None`        |
| `setSpringConstant(p_state, c_spring, idx_image=-1, idx_chain=-1)`                                 | `None`        |
| `setClimbingFalling(p_state, image_type, idx_image=-1, idx_chain=-1)`                              | `None`        |
| `setImageTypeAutomatically(p_state, idx_chain=-1)`                                                 | `None`        |

| Get GNEB Parameters                                                  | Returns       |
| -------------------------------------------------------------------- | ------------- |
| `getIterations(p_state, idx_chain=-1)`                               | `int, int`    |
| `getConvergence(p_state, idx_image=-1, idx_chain=-1)`                | `float`       |
| `getSpringConstant(p_state,  idx_image=-1, idx_chain=-1)`            | `float`       |
| `getClimbingFalling(p_state, idx_image=-1, idx_chain=-1)`            | `int`         |
| `getEnergyInterpolations(p_state, idx_chain=-1)`                     | `int`         |


Quantities
----------

| Get Physical Quantities                                              | Returns       |
| -------------------------------------------------------------------- | ------------- |
| `Get_Magnetization(p_state, idx_image=-1, idx_chain=-1)`             | `[3*float]` |

Simulation
----------

The available `method_type`s are:

| Method                        | Argument |
| ----------------------------- | :------: |
| Landau-Lifshitz-Gilbert       | `"LLG"`  |
| Geodesic Nudged Elastic Band  | `"GNEB"` |
| Monte-Carlo                   | `"MC"`   |

The available `solver_type`s are:

| Solver                        | Argument    |
| ----------------------------- | :---------: |
| Semi-Implicit Method B        | `"SIB"`     |
| Heun Method                   | `"Heun"`    |
| Depondt Method                | `"Depondt"` |
| Velocity Projection           | `"VP"`      |
| Nonlinear Conjugate Gradient  | `"NCG"`     |

Note that the VP and NCG Solvers are only meant for direct minimization and not for dynamics.

| Simulation state                                                                                                          | Returns    |
| ------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `SingleShot(p_state, method_type, solver_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1)`         | `None`     |
| `PlayPause(p_state, method_type, solver_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1)`          | `None`     |
| `Stop_All(p_state)`                                                                                                       | `None`     |
| `Running_Image(p_state, idx_image=-1, idx_chain=-1)`                                                                      | `Boolean`  |
| `Running_Chain(p_state, idx_chain=-1)`                                                                                    | `Boolean`  |
| `Running_Collection(p_state)`                                                                                             | `Boolean`  |
| `Running_Anywhere_Chain(p_state, idx_chain=-1)`                                                                           | `Boolean`  |
| `Running_Anywhere_Collection(p_state)`                                                                                    | `Boolean`  |


Transition
----------

| Transition options                                                           | Returns  | Description                                                                        |
| ---------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `Homogeneous(p_state, idx_1, idx_2, idx_chain=-1)`                           | `None`   | Generate homogeneous transition between two images of a chain                      |
| `Add_Noise_Temperature(p_state, temperature, idx_1, idx_2, idx_chain=-1)`    | `None`   | Add some temperature-scaled noise to a transition between two images of a chain    |


Input/Output
------------

| Macros of File Formats for Vector Fields | values  | Description                                       |
| ---------------------------------------- | :-----: | --------------------------------------------------|
| `IO_Fileformat_Regular`                  | 0       | sx sy sz (separated by whitespace)                |
| `IO_Fileformat_Regular_Pos`              | 1       | px py pz sx sy sz (separated by whitespace)       |
| `IO_Fileformat_CSV`                      | 2       | sx, sy, sz (separated by commas)                  |
| `IO_Fileformat_CSV_Pos`                  | 3       | px, py, pz, sx, sy, (sz separated by commas)      |
| `IO_Fileformat_OVF_bin8`                 | 4       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format |
| `IO_Fileformat_OVF_text`                 | 6       |                                                   |


| For Image                                                                                 | Description                          |
| ----------------------------------------------------------------------------------------- | ------------------------------------ |
| `Image_Read(p_state, filename, fileformat=0, idx_image=-1, idx_chain=-1)`                 | Read an image from disk              |
| `Image_Write(p_state, filename, fileformat=0, comment=" ", idx_image=-1, idx_chain=-1)`   | Write an image to disk               |
| `Image_Append(p_state, filename, fileformat=0, comment=" ", idx_image=-1, idx_chain=-1)`  | Append an image to an existing file  |

| For Chain                                                                     | Description                          |
| ----------------------------------------------------------------------------- | ------------------------------------ |
| `Chain_Read(p_state, filename, fileformat=0, idx_chain=-1)`                   | Read a chain of images from disk     |
| `Chain_Write(p_state, filename, fileformat=0, comment=" ", idx_chain=-1)`     | Write a chain of images to disk      |
| `Chain_Append(p_state, filename, fileformat=0, comment=" ", idx_chain=-1)`    | Append a chain of images to disk      |


---

[Home](README.md)