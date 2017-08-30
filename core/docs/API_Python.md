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
| `Get_NOI(p_state, idx_chain=-1)  `                                          | `int`          | Get Chain number of images                                 |
| `Get_Rx(p_state, idx_chain=-1)  `                                           | `Array`        | Get Rx                                                     |
| `Get_Rx_Interpolated(p_state, idx_chain=-1)  `                              | `array(float)` | Get Rx interpolated                                        |
| `Get_Energy(p_state, idx_chain=-1)  `                                       | `array(float)` | Get Energy of every System in Chain                        |
| `Get_Energy_Interpolated(p_state, idx_chain=-1)  `                          | `array(float)` | Get interpolated Energy of every System in Chain           |

| Image Manipulation                                              | Returns        | Description                                                |
| --------------------------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `Next_Image(p_state, idx_chain=-1)  `                           | `None`         | Switch active to next image of chain (one with largest index). If the current active is the last there is no effect. |
| `Prev_Image(p_state, idx_chain=-1)  `                           | `None`         | Switch active to previous image of chain (one with smaller index). If the current active is the frist one there is no effect |
| `Jump_To_Image(p_state, idx_image=-1, idx_chain=-1)  `          | `None`         | Switch active to specific image of chain. If this image does not exist there is no effect.          |
| `Image_to_Clipboard(p_state, idx_image=-1, idx_chain=-1)  `     | `None`         | Copy active image to clipboard                             |
| `Replace_Image(p_state, idx_image=-1, idx_chain=-1)  `          | `None`         | Replace active image in chain. If the image does not exist there is no effect.                      |
| `Insert_Image_Before(p_state, idx_image=-1, idx_chain=-1)  `    | `None`         | Inserts clipboard image before the current active image. Active image index is increment by one.    |
| `Insert_Image_After(p_state, idx_image=-1, idx_chain=-1)  `     | `None`         | Insert clipboard image after the current active image. Active image has the same index.             |
| `Push_Back(p_state, idx_chain=-1)  `                            | `None`         | Insert clipboard image at end of chain (after the image with the largest index).                    |
| `Delete_Image(p_state, idx_image=-1, idx_chain=-1)  `           | `None`         | Delete active image. If index is specified delete the corresponding image. If the image does not exist there is no effect. |
| `Pop_Back(p_state, idx_chain=-1)  `                             | `None`         | Delete image at end of chain.    |

| Data                                        | Returns        | Description                                                |
| ------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `Update_Data(p_state, idx_chain=-1)  `      | `None`         | Update the chain's data (interpolated energies etc.)       |
| `Setup_Data(p_state, idx_chain=-1)  `       | `None`         | Setup the chain's data arrays (when is this necessary?)    |


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

| Get Geometry parameters                                                         | Returns               | Description                                        |
| ------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `Get_Bounds(p_state, min, max, idx_image=-1, idx_chain=-1)  `                   | `None`                | Get Bounds                                         |
| `Get_Center(p_state, center, idx_image=-1, idx_chain=-1)  `                     | `None`                | Get Center                                         |
| `Get_Basis_Vectors(p_state, a, b, c, idx_image=-1, idx_chain=-1)  `             | `None`                | Get Basis vectors                                  |
| `Get_N_Cells(p_state, idx_image=-1, idx_chain=-1)  `                            | `Int, Int, Int`       | Get N Cells                                        |
| `Get_Translation_Vectors(p_state, ta, tb, tc, idx_image=-1, idx_chain=-1)  `    | `None`                | Get Translation Vectors                            |
| `Get_Dimensionality(p_state, idx_image=-1, idx_chain=-1)  `                     | `int`                 | Get Translation Vectors                            |
| `Get_Spin_Positions(p_state, idx_image=-1, idx_chain=-1)  `                     | `Array`               | Get Pointer to Spin Positions                      |


Hamiltonian
-----------

| Set Parameters                                                                  | Returns               | Description                                        |
| ------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `Set_Field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)  `        | `None`                | Set external magnetic field                        |
| `Set_Anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)  `   | `None`                | Set anisotropy                                     |
| `Set_STT(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)  `          | `None`                | Set spin transfer torque                           |
| `Set_Temperature(p_state, temperature, idx_image=-1, idx_chain=-1)  `           | `None`                | Set temperature                                    |


Log
---

| Log manipulation                                                         | Returns   | Description                 |
| ------------------------------------------------------------------------ | --------- | --------------------------- |
| `Send(p_state, level, sender, message, idx_image=-1, idx_chain=-1)  `    | `None`    | Send a Log message          |
| `Append(p_state)`                                                        | `None`    | Append Log to file          |


Parameters
----------

### LLG

| Set LLG Parameters                                                                            | Returns       |
| --------------------------------------------------------------------------------------------- | ------------- |
| `Set_LLG_N_Iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`   | `None`        |
| `Set_LLG_Direct_Minimization(p_state, use_minimization, idx_image=-1, idx_chain=-1)`          | `None`        |
| `Set_LLG_Convergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                       | `None`        |
| `Set_LLG_Time_Step(p_state, dt, idx_image=-1, idx_chain=-1)`                                  | `None`        |
| `Set_LLG_Damping(p_state, damping, idx_image=-1, idx_chain=-1)`                               | `None`        |
| `Set_LLG_STT(p_state, use_gradient, magnitude, direction, idx_image=-1, idx_chain=-1)`        | `None`        |
| `Set_LLG_Temperature(p_state, temperature, idx_image=-1, idx_chain=-1)`                       | `None`        |

| Get LLG Parameters                                                     | Returns       |
| ---------------------------------------------------------------------- | ------------- |
| `Get_LLG_N_Iterations(p_state, idx_image=-1, idx_chain=-1)`            | `int, int`    |
| `Get_LLG_Direct_Minimization(p_state, idx_image=-1, idx_chain=-1)`     | `int`         |
| `Get_LLG_Convergence(p_state, idx_image=-1, idx_chain=-1)`             | `float`       |
| `Get_LLG_Time_Step(p_state, idx_image=-1, idx_chain=-1)`               | `float`       |
| `Get_LLG_Damping(p_state, idx_image=-1, idx_chain=-1)`                 | `float`       |
| `Get_LLG_STT(p_state, idx_image=-1, idx_chain=-1)`                     | `float, [3*float], bool` |
| `Get_LLG_Temperature(p_state, idx_image=-1, idx_chain=-1)`             | `float`       |

### GNEB

| Set GNEB Parameters                                                                                | Returns       |
| -------------------------------------------------------------------------------------------------- | ------------- |
| `Set_GNEB_N_Iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`       | `None`        |
| `Set_GNEB_Convergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                           | `None`        |
| `Set_GNEB_Spring_Constant(p_state, c_spring, idx_image=-1, idx_chain=-1)`                          | `None`        |
| `Set_GNEB_Climbing_Falling(p_state, image_type, idx_image=-1, idx_chain=-1)`                       | `None`        |
| `Set_GNEB_Image_Type_Automatically(p_state, idx_chain=-1)`                                         | `None`        |

| Get GNEB Parameters                                                  | Returns       |
| -------------------------------------------------------------------- | ------------- |
| `Get_GNEB_N_Iterations(p_state, idx_chain=-1)`                       | `int, int`    |
| `Get_GNEB_Convergence(p_state, idx_image=-1, idx_chain=-1)`          | `float`       |
| `Get_GNEB_Spring_Constant(p_state,  idx_image=-1, idx_chain=-1)`     | `float`       |
| `Get_GNEB_Climbing_Falling(p_state, idx_image=-1, idx_chain=-1)`     | `int`         |
| `Get_GNEB_N_Energy_Interpolations(p_state, idx_chain=-1)`            | `int`         |


Simulation
----------

The available `method_type`s are:

| Method                        | Argument |
| ----------------------------- | :------: |
| Landau-Lifshitz-Gilbert       | `"LLG"`  |
| Geodesic Nudged Elastic Band  | `"GNEB"` |
| Monte-Carlo                   | `"MC"`   |

The available `optimizer_type`s are:

| optimizer                     | Argument |
| ----------------------------- | :------: |
| Semi-Implicit Method          | `"SIB"`  |
| Velocity Projection           | `"VP"`   |
| Heun Method (Direct)          | `"Heun"` |
| Nonlinear Conjugate Gradient  | `"NCG"`  |


| Simulation state                                                                                                          | Returns    |
| ------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `SingleShot(p_state, method_type, optimizer_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1)`      | `None`     |
| `PlayPause(p_state, method_type, optimizer_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1)`       | `None`     |
| `Stop_All(p_state )`                                                                                                      | `None`     |
| `Running_LLG(p_state, idx_image=-1, idx_chain=-1)`                                                                        | `Boolean`  |
| `Running_LLG_Chain(p_state, idx_chain=-1)`                                                                                | `Boolean`  |
| `Running_LLG_Anywhere(p_state )`                                                                                          | `Boolean`  |
| `Running_GNEB(p_state, idx_chain=-1)`                                                                                     | `Boolean`  |
| `Running_GNEB_Anywhere(p_state )`                                                                                         | `Boolean`  |
| `Running_MMF(p_state )`                                                                                                   | `Boolean`  |
| `Running_Any(p_state, idx_image=-1, idx_chain=-1)`                                                                        | `Boolean`  |
| `Running_Any_Anywhere(p_state )`                                                                                          | `Boolean`  |


Transition
----------

| Transition options                                                           | Returns  | Description                                                                        |
| ---------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `Homogeneous(p_state, idx_1, idx_2, idx_chain=-1)  `                         | `None`   | Generate homogeneous transition between two images of a chain                      |
| `Add_Noise_Temperature(p_state, temperature, idx_1, idx_2, idx_chain=-1)`    | `None`   | Add some temperature-scaled noise to a transition between two images of a chain    |


---

[Home](Readme.md)