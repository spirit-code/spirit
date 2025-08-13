Spirit inputfile (TOML)
====================================================

:::{versionchanged} 3.0.0

*For the deprecated input specification, see:* [Spirit inputfile (Legacy)](/core/docs/Input_Legacy.md)

To ease the transition the [python package]() provides a CLI utiliy to convert to your configuration files to the new format.
Run `spirit-cfgconvert --help` to learn how to use it.
:::

Introduction
----------------------------------------------------

The input file uses the standardized [TOML](https://toml.io) format.
Its structure is hierarchical in that every entry is identified by a path.
Each level or section in the path is separated by a `.` character and key-value pairs are specified with an assignment expression. e.g:
```toml
geometry.lattice_constant = 1.41
geometry.bravais_lattice = "sc"
```
Paths that share a common prefix can be grouped into sections.
A section begins with the common prefix in braces e.g., `[geometry]`, so the above example is equivalent to:
```toml
[geometry]
lattice_constant = 1.41
bravais_lattice = "sc"
```

Any assignment expression following a section specifier belongs to that section until a new section is declared.
For details, please refer to the [TOML specification](https://toml.io).

### Sections

- [`[defaults]` - General settings](#general-settings-defaults)
- [`[logging]` - Logging](#logging)
- [`[geometry]` - Geometry](#geometry)
    - [Pinning](#pinning)
    - [Disorder and Defects](#disorder-and-defects)
- [`[hamiltonian]` - Hamiltonian](/core/docs/Hamiltonian.md)
- [`[method]` Method Configurations](#method-configuration)
    - [Method Output](#method-output)
    - [Method Parameters](#method-parameters)


---


### Tables

Many keywords in the configuration file support tables e.g., the pair interactions:
```toml
pairs = """
i j   da db dc    Jij   Dij  Dijx Dijy Dijz
0 0    1  0  0   10.0   6.0   1.0  0.0  0.0
0 0    0  1  0   10.0   6.0   0.0  1.0  0.0
0 0    0  0  1   10.0   6.0   0.0  0.0  1.0
"""
```

These can usually also be read from an external file by providing a path with the `fils://` prefix:
```toml
pairs = "file://input/pairs.txt"
```
where the contents of `input/pairs.txt` looks like this:
```text
i j   da db dc    Jij   Dij  Dijx Dijy Dijz
0 0    1  0  0   10.0   6.0   1.0  0.0  0.0
0 0    0  1  0   10.0   6.0   0.0  1.0  0.0
0 0    0  0  1   10.0   6.0   0.0  0.0  1.0
```

:::{note}
It is important that the first line contains the header for the table.
:::


---


General Settings (Defaults)
----------------------------------------------------

*section:* `[defaults]`

```toml
### Add a tag to output files (for timestamp use "<time>")
[defaults.output]
file_tag        = "some_tag"
folder          = "output"
```


---


Logging
----------------------------------------------------

*section:* `[logging]`

```toml
[logging]
output.folder = "."
### Save input parameters on creation of State
input_save_initial = false
### Save input parameters on deletion of State
input_save_final   = false

### Print log messages to the console
log_to_console = true
### Print messages up to (including) log_console_level
console_level  = 5

### Save the log as a file
log_to_file = true
### Save messages up to (including) log_file_level
file_level  = 5
```

Except for `SEVERE` and `ERROR`, only log messages up to
`log_console_level` will be printed and only messages up to
`log_file_level` will be saved.
If `log_to_file`, however is set to zero, no file is written
at all.

| Log Levels | Integer | Description            |
| ---------- | ------- | ---------------------- |
| ALL        |    0    | Everything             |
| SEVERE     |    1    | Only severe errors     |
| ERROR      |    2    | Also non-fatal errors  |
| WARNING    |    3    | Also warnings          |
| PARAMETER  |    4    | Also input parameters  |
| INFO       |    5    | Also info-messages     |
| DEBUG      |    6    | Also deeper debug-info |


---


Geometry
----------------------------------------------------

*section*: `[geometry]`

The Geometry of a spin system is specified in form of a bravais lattice
and a basis cell of atoms. The number of basis cells along each principal
direction of the basis can be specified.
*Note:* the default basis is a single atom at (0,0,0).

**3D simple cubic example:**

```toml
### The bravais lattice type
bravais_lattice = "sc"

### µSpin
mu_s = 2.0

### Number of basis cells along principal
### directions (a b c)
n_basis_cells = [100, 100, 10]
```

If you have a nontrivial basis cell, note that you should specify `mu_s`
for all atoms in your basis cell (see the next example).

**2D honeycomb example:**

```toml
### The bravais lattice type
bravais_lattice = "hex2d"

### The basis cell in units of bravais vectors
### n            No of spins in the basis cell
### 1.x 1.y 1.z  position of spins within basis
### 2.x 2.y 2.z  cell in terms of bravais vectors
basis = """
0          0         0
0.33333333 0.3333333 0
"""
### µSpin
mu_s = [2.0, 1.0]

### Number of basis cells along principal
### directions (a b c)
n_basis_cells = [100, 100, 1]
```

The builtin bravais lattice types are the following:

| Bravais Lattice Type     | Keyword  | Comment                     |
| ------------------------ | -------- | --------------------------- |
| Simple cubic             | sc       |                             |
| Body-centered cubic      | bcc      |                             |
| Face-centered cubic      | fcc      |                             |
| Hexagonal (2D)           | hex2d    |  60deg angle                |
| Hexagonal (2D)           | hex2d60  |  60deg angle                |
| Hexagonal (2D)           | hex2d120 | 120deg angle                |

All crystal structures can be specified manually through a basis and
a set of bravais vectors:

```toml
### bravais_vectors or bravais_matrix
###   a.x a.y a.z       a.x b.x c.x
###   b.x b.y b.z       a.y b.y c.y
###   c.x c.y c.z       a.z b.z c.z
bravais_vectors = """
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
"""
```

A lattice constant can be used for scaling:
```toml
### Scaling constant
lattice_constant = 1.0
```
Note that it scales the Bravais vectors and therefore the
translations, atom positions in the basis cell and potentially
-- if you specified them in terms of the Bravais vectors --
also the anisotropy and DM vectors.

**Units:**

The Bravais vectors (or matrix) are specified in Cartesian coordinates in units of Angstrom.
The basis atoms are specified in units of the Bravais vectors.

The atomic moments `mu_s` are specified in units of the Bohr magneton `mu_B`.

Boundary conditions are specified as a triplet of integers, each number's position corresponds to the dimension in which the boundary condition applies.

```toml
### Boundary_conditions (a b c): 0(open), 1(periodical)
boundary_conditions       = [1, 1, 0]
```


### Pinning

*section*: `[geometry]`

Note that for this feature you need to build with `SPIRIT_ENABLE_PINNING`
set to `ON` in cmake.

When pinning the boundary you have to specify how many columns, rows and layers
of cells should be pinned. That means `pinning.boundary` has to be an array of
length 3 where each entry is either a number for symmetric counts or a pair of values.
To set the direction of the pinned cells, you need to give the `pinning.cell`
keyword and one vector for each basis atom.

You can for example do the following to create a U-shaped pinning in x-direction:
```toml
[geometry]
pinning.boundary = [
 [2, 0], # Pin left side of the sample (2 rows)
 2,      # Pin top and bottom sides (2 rows each)
 0
]
# Pin the atoms to x-direction
pinning.cell = [
  [1, 0, 0]
]
```

To specify individual pinned sites (overriding the above pinning settings),
insert a [table](#tables) into your input. For example:
```toml
### Specify the number of pinned sites and then the sites (in terms of translations) and directions
pinned = """
i  da db dc     x   y   z
0   0  0  0   1.0 0.0 0.0
0   1  0  0   0.0 1.0 0.0
0   0  1  0   0.0 0.0 1.0
"""
```
You may also place it into a separate file with the `file://` prefix, e.g.
```toml
### Read pinned sites from a separate file
pinned = "file://input/pinned.txt"
```
The file should either contain only the pinned sites or you need to specify `n_pinned`
inside the file.


### Disorder and Defects

*section*: `[geometry]`

Note that for this feature you need to build with `SPIRIT_ENABLE_DEFECTS`
set to `ON` in cmake.

In order to specify disorder across the lattice, you can write for example a
single atom basis with 50% chance of containing one of two atom types (0 or 1) in terms of the [table](#tables):
```toml
# iatom  atom_type  concentration   mu_s  ...
atom_types = """
i type    c    mu_s
0    1  2.0     0.5
"""
```

Note that you have to also specify the magnetic moment, as this is now site-
and atom type dependent.

A two-atom basis where
- the first atom is type 0
- the second atom is 70% type 1 and 30% type 2
```toml
# iatom  atom_type  concentration  mu_s
atom_types = """
i type     c  mu_s
0    0     1   1.0
1    1   0.7   2.5
1    2   0.3   2.3
"""
```
The total concentration on a site should not be more than `1`. If it is less
than `1`, vacancies will appear.

To specify defects, be it vacancies or impurities, you may fix atom types for
sites of the whole lattice by inserting a [table](#tables) into your input. For example:
```toml
### Atom types: type index 0..n or or vacancy (type < 0)
### Specify the number of defects and then the defects in terms of translations and type
### i  da db dc  itype
defects = """
i da db dc type
0  0 0 0    -1
0  1 0 0    -1
0  0 1 0    -1
"""
```
You may also place it into a separate file with the `file://` prefix,
e.g.
```toml
[geometry]
### Read defects from a separate file
defects_from = "file://input/defects.txt"
```
The file should either contain only the defects or you need to specify `n_defects`
inside the file.


---


Hamiltonian
----------------------------------------------------

*section:* `[hamiltonian]`

:::{attention}
 The configuration of the interactions is documented [here](/core/docs/Hamiltonian.md).
:::

---


Method Configuration
-------------------------------------------------------------------

### Method Output

*sections:* `[method."<method>".output]`

For `llg` and equivalently `mc` and `gneb`, you can specify which
output you want your simulations to create.
These are specified in the `output` section under within method section.
They share a few common output types, for example:

```toml
[method."<method>".output]
any     = true    # Write any output at all
initial = true    # Save before the first iteration
final   = true    # Save after the last iteration
```

Note in the following that `step` means after each `N` iterations and
denotes a separate file for each step, whereas `archive` denotes that
results are appended to an archive file at each step.

The energy output files are in units of meV, and can be switched to
meV per spin with `method.<method>.output.energy_divide_by_nspins`.

**LLG:**
```toml
[method.llg.output]
energy_step             = false    # Save system energy at each step
energy_archive          = true     # Archive system energy at each step
energy_spin_resolved    = false    # Also save energies for each spin
energy_divide_by_nspins = true     # Normalize energies with number of spins

configuration_step      = true     # Save spin configuration at each step
configuration_archive   = false    # Archive spin configuration at each step
```

**MC:**
```toml
[method.mc.output]
energy_step             = false
energy_archive          = true
energy_spin_resolved    = false
energy_divide_by_nspins = true

configuration_step    = true
configuration_archive = false
```

**GNEB:**
```toml
[method.gneb.output]
energies_step             = false # Save energies of images in chain
energies_interpolated     = true  # Also save interpolated energies
energies_divide_by_nspins = true  # Normalize energies with number of spins

chain_step = false    # Save the whole chain at each step
```


### Method Parameters

*sections:* `[method."<method>"]`

Again, the different Methods share a few common parameters.

```toml
[method."<method>"]
### Maximum wall time for single simulation
### hh:mm:ss, where 0:0:0 is infinity
max_walltime       = "0:0:0"

### Force convergence parameter
force_convergence  = 10e-9

### Number of iterations
n_iterations       = 2000000
### Number of iterations after which to save
n_iterations_log   = 2000
### Number of iterations that gets run with no checks or outputs (Increasing this boosts performance, especially in CUDA builds)
n_iterations_amortize = 1

```

**LLG:**

```toml
[method.llg]
### Seed for Random Number Generator
seed            = 20006

### Damping [none]
damping         = 0.3

### Time step dt [ps]
dt              = 1.0e-3

### Temperature [K]
temperature                      = 0
temperature_gradient.magnitude   = 0
temperature_gradient.direction   = [1, 0, 0]

### Spin current model:
### 'gradient':  spin-orbit torque
### 'monolayer': spin-transfer torque
spin_current_model            = "monolayer"
### Spin current vector:
### proportional to the injected current density
spin_current_vector.magnitude = 0.0
spin_current_vector.direction = [1.0, 0.0, 0.0]
```

The time step `dt` is given in picoseconds.
The temperature is given in Kelvin and the temperature gradient in Kelvin/Angstrom.

If you don't specify a seed for the RNG, it will be chosen randomly.

**MC:**

```toml
[method.mc]
### Seed for Random Number Generator
seed             = 20006

### Temperature [K]
temperature      = 0

### Acceptance ratio
acceptance_ratio = 0.5
```

The temperature is given in Kelvin.

If you don't specify a seed for the RNG, it will be chosen randomly.

**GNEB:**

```toml
[method.gneb]
### Constant for the spring force
spring_constant = 1.0

### Number of energy interpolations between images
n_energy_interpolations = 10
```



---

[Home](Readme.md)
