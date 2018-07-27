SPIRIT Python API
====================

State
-----

To create a new state with one chain containing a single image, initialized by an [input file](INPUT.md), and run the most simple example of a **spin dynamics simulation**:
```python
from spirit import state
from spirit import simulation

cfgfile = "input/input.cfg"           # Input File
with state.State(cfgfile) as p_state: # State setup
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB) # Start a LLG simulation using the SIB solver
```
or call setup and delete manually:
```python
from spirit import state
from spirit import simulation

cfgfile = "input/input.cfg"    # Input File
p_state = state.setup(cfgfile) # State setup
simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB) # Start a LLG simulation using the SIB solver
state.delete(p_state)          # State cleanup
```

You can pass a [config file](INPUT.md) specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
**Note that you currently cannot change the geometry of the systems in your state once they are initialized.**

| State manipulation                                                                  | Returns    |
| ----------------------------------------------------------------------------------- | ---------- |
| `setup( configfile="", quiet=False )`                                               | `None`     |
| `delete(p_state )`                                                                  | `None`     |


System
------

| System                                                                | Returns  | Description                                                                          |
| --------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------ |
| `get_index(p_state)`                                                  | `int`    | Returns the index of the currently active image                                      |
| `get_nos(p_state, idx_image=-1, idx_chain=-1)`                        | `int`    | Returns the number of spins                                                          |
| `get_spin_directions(p_state, idx_image=-1, idx_chain=-1)`            | `[3*NOS]`| Returns an `numpy.Array` of size `3*NOS` with the components of each spin's vector   |
| `get_energy(p_state, idx_image=-1, idx_chain=-1)`                     | `float`  | Returns the energy of the system                                                     |
| `update_data(p_state, idx_image=-1, idx_chain=-1)`                    | `None`   | Update the data of the state                                                         |
| `print_energy_array(p_state, idx_image=-1, idx_chain=-1)`             | `None`   | Print the energy array of the state                                                  |


Chain
-----

For having more images one can copy the active image in the Clipboard and then insert in a specified position of the chain.
```python
chain.image_to_clipboard(p_state )   # Copy p_state to Clipboard
chain.insert_image_after(p_state )   # Insert the image from Clipboard right after the currently active image
```
For getting the total number of images in the chain
```python
number_of_images = chain.get_noi(p_state )
```

| Get Info                                                                    | Returns        | Description                                                |
| --------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `get_index(p_state )`                                                       | `int`          | Get Chain index                                            |
| `get_noi(p_state, idx_chain=-1)`                                            | `int`          | Get Chain number of images                                 |
| `get_rx(p_state, idx_chain=-1)`                                             | `Array`        | Get Rx                                                     |
| `get_rx_interpolated(p_state, idx_chain=-1)`                                | `Array(float)` | Get Rx interpolated                                        |
| `get_energy(p_state, idx_chain=-1)`                                         | `Array(float)` | Get Energy of every System in Chain                        |
| `get_energy_interpolated(p_state, idx_chain=-1)`                            | `Array(float)` | Get interpolated Energy of every System in Chain           |

| Image Manipulation                                              | Returns        | Description                                                |
| --------------------------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `next_image(p_state, idx_chain=-1)`                             | `None`         | Switch active to next image of chain (one with largest index). If the current active is the last there is no effect. |
| `prev_image(p_state, idx_chain=-1)`                             | `None`         | Switch active to previous image of chain (one with smaller index). If the current active is the first one there is no effect |
| `jump_to_image(p_state, idx_image=-1, idx_chain=-1)`            | `None`         | Switch active to specific image of chain. If this image does not exist there is no effect.          |
| `image_to_clipboard(p_state, idx_image=-1, idx_chain=-1)`       | `None`         | Copy active image to clipboard                             |
| `replace_image(p_state, idx_image=-1, idx_chain=-1)`            | `None`         | Replace active image in chain. If the image does not exist there is no effect.                      |
| `insert_image_before(p_state, idx_image=-1, idx_chain=-1)`      | `None`         | Inserts clipboard image before the current active image. Active image index is increment by one.    |
| `insert_image_after(p_state, idx_image=-1, idx_chain=-1)`       | `None`         | Insert clipboard image after the current active image. Active image has the same index.             |
| `push_back(p_state, idx_chain=-1)`                              | `None`         | Insert clipboard image at end of chain (after the image with the largest index).                    |
| `delete_image(p_state, idx_image=-1, idx_chain=-1)`             | `None`         | Delete active image. If index is specified delete the corresponding image. If the image does not exist there is no effect. |
| `pop_back(p_state, idx_chain=-1)`                               | `None`         | Delete image at end of chain.    |

| Data                                        | Returns        | Description                                                |
| ------------------------------------------- | -------------- | ---------------------------------------------------------- |
| `update_data(p_state, idx_chain=-1)`        | `None`         | Update the chain's data (interpolated energies etc.)       |
| `setup_data(p_state, idx_chain=-1)`         | `None`         | Setup the chain's data arrays                              |


Constants
---------

| Physical Constants                      | Returns        | Description                                        |
| --------------------------------------- | -------------- | -------------------------------------------------- |
| `mu_B`                                  | `float`        | The Bohr Magneton [meV / T]                        |
| `k_B`                                   | `float`        | The Boltzmann constant [meV / K]                   |
| `hbar`                                  | `float`        | Planck's constant over 2pi [meV*ps / rad]          |
| `mRy`                                   | `float`        | Millirydberg [mRy / meV]                           |
| `gamma`                                 | `float`        | The Gyromagnetic ratio of electron [rad / (ps*T)]  |
| `g_e`                                   | `float`        | The Electron g-factor [unitless]                   |


Geometry
--------

| Set Geometry parameters                                                       | Description                                           |
| ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| `set_bravais_lattice_type(p_state, lattice_type, idx_image=-1, idx_chain=-1)` | Set the bravais vectors to a pre-defined lattice type |
| `set_n_cells(p_state, n_cells=[1, 1, 1], idx_image=-1, idx_chain=-1)`         | Set the number of basis cells along bravais vectors   |
| `set_mu_s(p_state, mu_s, idx_image=-1, idx_chain=-1)`                         | Set the magnetic moment of all atoms                  |
| `set_cell_atom_types(p_state, atom_types, idx_image=-1, idx_chain=-1)`        | Set the atom types of the cell atoms                  |
| `set_bravais_vectors(p_state, ta=[1.0, 0.0, 0.0], tb=[0.0, 1.0, 0.0], tc=[0.0, 0.0, 1.0], idx_image=-1, idx_chain=-1)` | Manually specify bravais vectors |
| `set_lattice_constant(p_state, lattice_constant, idx_image=-1, idx_chain=-1)` | Set the global lattice constant                       |

| Get Geometry parameters                                              | Returns               | Description                                        |
| -------------------------------------------------------------------- | --------------------- | -------------------------------------------------- |
| `get_bounds(p_state, idx_image=-1, idx_chain=-1)`                    | `[3], [3]`            | Get bounds (minimum and maximum arrays)            |
| `get_center(p_state, idx_image=-1, idx_chain=-1)`                    | `float, float, float` | Get center                                         |
| `get_basis_vectors(p_state, idx_image=-1, idx_chain=-1)`             | `[3],[3],[3]`         | Get basis vectors                                  |
| `get_n_cells(p_state, idx_image=-1, idx_chain=-1)`                   | `Int, Int, Int`       | Get number of  cells in each dimension             |
| `get_translation_vectors(p_state, idx_image=-1, idx_chain=-1)`       | `[3],[3],[3]`         | Get translation vectors                            |
| `get_dimensionality(p_state, idx_image=-1, idx_chain=-1)`            | `int`                 | Get dimensionality of the system                   |
| `get_spin_positions(p_state, idx_image=-1, idx_chain=-1)`            | `[3*NOS]`             | Get Spin positions                                 |
| `get_atom_types(p_state, idx_image=-1, idx_chain=-1)`                | `[NOS]`               | Get atom types                                     |


Hamiltonian
-----------

| Set Parameters                                                                  | Returns               | Description                                                 |
| ------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------- |
| `set_boundary_conditions(p_state, boundaries, idx_image=-1, idx_chain=-1)`      | `None`                | Set the boundary conditions [a, b, c]: 0=open, 1=periodical |
| `set_field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)`          | `None`                | Set external magnetic field                                 |
| `set_anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1)`     | `None`                | Set a magnitude and normal of anisotropy for all spins      |
| `set_exchange(p_state, n_shells, J_ij, idx_image=-1, idx_chain=-1)`             | `None`                | Set the exchange pairs in terms of neighbour shells         |
| `set_dmi(p_state, n_shells, D_ij, idx_image=-1, idx_chain=-1)`                  | `None`                | Set the DMI pairs in terms of neighbour shells              |
| `set_ddi(p_state, radius, idx_image=-1, idx_chain=-1)`                          | `None`                | Set dipole-dipole cutoff radius                             |


Log
---

| Log manipulation                                                         | Returns   | Description                 |
| ------------------------------------------------------------------------ | --------- | --------------------------- |
| `send(p_state, level, sender, message, idx_image=-1, idx_chain=-1)`      | `None`    | Send a Log message          |
| `append(p_state)`                                                        | `None`    | Append Log to file          |


Parameters
----------

### LLG

| Set LLG Parameters                                                                            | Returns       |
| --------------------------------------------------------------------------------------------- | ------------- |
| `set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`          | `None`        |
| `set_direct_minimization(p_state, use_minimization, idx_image=-1, idx_chain=-1)`                | `None`        |
| `set_convergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                            | `None`        |
| `set_time_step(p_state, dt, idx_image=-1, idx_chain=-1)`                                        | `None`        |
| `set_damping(p_state, damping, idx_image=-1, idx_chain=-1)`                                    | `None`        |
| `set_stt(p_state, use_gradient, magnitude, direction, idx_image=-1, idx_chain=-1)`             | `None`        |
| `set_temperature(p_state, temperature, idx_image=-1, idx_chain=-1)`                            | `None`        |

| Get LLG Parameters                                                     | Returns       |
| ---------------------------------------------------------------------- | ------------- |
| `get_iterations(p_state, idx_image=-1, idx_chain=-1)`                   | `int, int`    |
| `get_direct_minimization(p_state, idx_image=-1, idx_chain=-1)`          | `int`         |
| `get_convergence(p_state, idx_image=-1, idx_chain=-1)`                  | `float`       |
| `get_time_step(p_state, idx_image=-1, idx_chain=-1)`                     | `float`       |
| `get_damping(p_state, idx_image=-1, idx_chain=-1)`                      | `float`       |
| `get_stt(p_state, idx_image=-1, idx_chain=-1)`                          | `float, [3], bool` |
| `get_temperature(p_state, idx_image=-1, idx_chain=-1)`                  | `float`       |

### GNEB

| Set GNEB Parameters                                                                                | Returns       |
| -------------------------------------------------------------------------------------------------- | ------------- |
| `set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1)`               | `None`        |
| `set_convergence(p_state, convergence, idx_image=-1, idx_chain=-1)`                                 | `None`        |
| `set_spring_constant(p_state, c_spring, idx_image=-1, idx_chain=-1)`                                 | `None`        |
| `set_climbing_falling(p_state, image_type, idx_image=-1, idx_chain=-1)`                              | `None`        |
| `set_image_type_automatically(p_state, idx_chain=-1)`                                                 | `None`        |

| Get GNEB Parameters                                                  | Returns       |
| -------------------------------------------------------------------- | ------------- |
| `get_iterations(p_state, idx_chain=-1)`                               | `int, int`    |
| `get_convergence(p_state, idx_image=-1, idx_chain=-1)`                | `float`       |
| `get_spring_constant(p_state,  idx_image=-1, idx_chain=-1)`            | `float`       |
| `get_climbing_falling(p_state, idx_image=-1, idx_chain=-1)`            | `int`         |
| `get_energy_interpolations(p_state, idx_chain=-1)`                     | `int`         |


Quantities
----------

| Get Physical Quantities                                              | Returns       |
| -------------------------------------------------------------------- | ------------- |
| `get_magnetization(p_state, idx_image=-1, idx_chain=-1)`             | `[3*float]` |

Simulation
----------

The available `method_type`s are:

| Method                        | Argument     |
| ----------------------------- | :----------: |
| Monte-Carlo                   | `METHOD_MC`  |
| Landau-Lifshitz-Gilbert       | `METHOD_LLG` |
| Geodesic Nudged Elastic Band  | `METHOD_GNEB`|
| Mode Following Method         | `METHOD_MMF` |
| Eigenmode analysis            | `METHOD_EMA` |

The available `solver_type`s are:

| Solver                        | Argument      |
| ----------------------------- | :-----------: |
| Semi-Implicit Method B        | `SOLVER_SIB`  |
| Heun Method                   | `SOLVER_HEUN` |
| Depondt Method                | `SOLVER_HEUN` |
| Velocity Projection           | `SOLVER_VP`   |

Note that the VP and NCG Solvers are only meant for direct minimization and not for dynamics.

| Simulation state                                        | Returns    |
| ------------------------------------------------------- | ---------- |
| `start(p_state, method_type, solver_type=None, n_iterations=-1, n_iterations_log=-1, single_shot=False, idx_image=-1, idx_chain=-1)` | `None` |
| `single_shot(p_state, idx_image=-1, idx_chain=-1)`      | `None`     |
| `stop(p_state, idx_image=-1, idx_chain=-1)`             | `None`     |
| `stop_all(p_state)`                                     | `None`     |
| `running_on_image(p_state, idx_image=-1, idx_chain=-1)` | `Boolean`  |
| `running_on_chain(p_state, idx_chain=-1)`               | `Boolean`  |
| `running_anywhere_on_chain(p_state, idx_chain=-1)`      | `Boolean`  |


Transition
----------

| Transition options                                                           | Returns  | Description                                                                        |
| ---------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `homogeneous(p_state, idx_1, idx_2, idx_chain=-1)`            | `None`   | Generate homogeneous transition between two images of a chain                           |
| `add_noise(p_state, temperature, idx_1, idx_2, idx_chain=-1)` | `None`   | Add some temperature-scaled noise to a transition between two images of a chain    |


Input/Output
------------

Note that, when reading an image or chain from file, the file will automatically be tested for an OVF header.
If it cannot be identified as OVF, it will be tried to be read as three plain text columns (Sx Sy Sz).

Note also, IO is still being re-written and only OVF will be supported as output format.

| For Image                                                                                 | Description                          |
| ----------------------------------------------------------------------------------------- | ------------------------------------ |
| `n_images_in_file(p_state, filename, idx_image_inchain=-1, idx_chain=-1)`   | Read specified image from a file to specified image in the chain |
| `image_read(p_state, filename, idx_image_infile=0, idx_image_inchain=-1, idx_chain=-1)`   | Read specified image from a file to specified image in the chain |
| `image_write(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment=" ", idx_image=-1, idx_chain=-1)`   | Write an image to disk               |
| `image_append(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment=" ", idx_image=-1, idx_chain=-1)`  | Append an image to an existing file  |

| For Chain                                                                     | Description                          |
| ----------------------------------------------------------------------------- | ------------------------------------ |
| `chain_read(p_state, filename, starting_image=-1, ending_image=-1, insert_idx=-1, idx_chain=-1)` | Read some images from a file and insert them into the chain, starting at a specified index |
| `chain_write(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment=" ", idx_chain=-1)`     | Write a chain of images to disk      |
| `chain_append(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment=" ", idx_chain=-1)`    | Append a chain of images to disk     |

| Macros of File Formats for Vector Fields | values  | Description                                       |
| ---------------------------------------- | :-----: | --------------------------------------------------|
| `FILEFORMAT_OVF_BIN`                     | 0       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format (binary, automatically determined size)  |
| `FILEFORMAT_OVF_BIN4`                    | 1       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format (binary 4)  |
| `FILEFORMAT_OVF_BIN8`                    | 2       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format (binary 8)  |
| `FILEFORMAT_OVF_TEXT`                    | 3       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format (plaintext) |
| `FILEFORMAT_OVF_CSV`                     | 4       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format (comma-separated plaintext) |


---

[Home](README.md)