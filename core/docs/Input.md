Spirit inputfile
====================================================

The following sections will list and explain the input file keywords.

1. [General Settings and Log](#general-settings-and-log)
2. [Geometry](#geometry)
    1. [Pinning](#pinning)
    2. [Disorder and Defects](#disorder-and-defects)
3. [Heisenberg Hamiltonian](#heisenberg-hamiltonian)
4. [Method Configuration](#method-configuration)
    1. [Method Output](#method-output)
    2. [Method Parameters](#method-parameters)


General Settings and Log
----------------------------------------------------

```toml
### Add a tag to output files (for timestamp use "<time>")
[defaults.output]
file_tag        = "some_tag"
folder          = "output"
```

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


Geometry
----------------------------------------------------

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


### Pinning

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
insert a table into your input. For example:
```toml
[geometry]
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
[geometry]
### Read pinned sites from a separate file
pinned = "file://input/pinned.txt"
```
The file should either contain only the pinned sites or you need to specify `n_pinned`
inside the file.


### Disorder and Defects

Note that for this feature you need to build with `SPIRIT_ENABLE_DEFECTS`
set to `ON` in cmake.

In order to specify disorder across the lattice, you can write for example a
single atom basis with 50% chance of containing one of two atom types (0 or 1):
```toml
[geometry]
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
[geometry]
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
sites of the whole lattice by inserting a list into your input. For example:
```toml
[geometry]
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

Heisenberg Hamiltonian
----------------------------------------------------

To use a Heisenberg Hamiltonian, use either `heisenberg_neighbours` or `heisenberg_pairs`
as input parameter after the `hamiltonian` keyword.

The Hamiltonian is defined as

![](https://math.vercel.app/?bgcolor=auto&from=%0A%09%5Cmathcal%7BH%7D%20%3D%0A%20%20%20%20%20%20-%20%5Csum_i%20%5Cmu_i%20%5Cvec%7BB%7D%5Ccdot%5Cvec%7Bn%7D_i%0A%20%20%20%20%20%20%20-%20%5Csum_i%20%5Csum_j%20K_j%20%28%5Chat%7BK%7D_j%5Ccdot%5Cvec%7Bn%7D_i%29%5E2%5C%5C%0A%20%20%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20J_%7Bij%7D%20%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%0A%20%20%20%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20%5Cvec%7BD%7D_%7Bij%7D%20%5Ccdot%20(%5Cvec%7Bn%7D_i%5Ctimes%5Cvec%7Bn%7D_j)%5C%5C%0A%20%20%20%20%20%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cmu_0%7D%7B4%5Cpi%7D%20%5Csum_%7B%5Csubstack%7Bi%2Cj%20%5C%5C%20i%20%5Cneq%20j%7D%7D%20%5Cmu_i%20%5Cmu_j%20%5Cfrac%7B(%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Chat%7Br%7D_%7Bij%7D)%20(%5Cvec%7Bn%7D_j%5Ccdot%5Chat%7Br%7D_%7Bij%7D)%20-%20%5Cvec%7Bn%7D_i%20%5Cvec%7Bn%7D_j%7D%7B%7Br_%7Bij%7D%7D%5E3%7D)

where `<ij>` denotes the unique pairs of interacting spins `i` and `j`.
For more details, such as the notation used here, see [Phys. Rev. B **99** 224414 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.224414).

**General Parameters:**

```toml
### Boundary conditions (in a b c) = 0(open), 1(periodical)
boundary_conditions      = [1, 1, 0]

### External magnetic field [T]
external_field.magnitude = 25.0
external_field.direction = [0.0, 0.0, 1.0]

### Uniaxial anisotropy constant [meV]
anisotropy.magnitude     = 0.0
anisotropy.normal        = [0.0, 0.0, 1.0]

### Dipole-dipole interaction caclulation method
### (none, fft, fmm, cutoff)
ddi_method               = 'fft'

### DDI number of periodic images (fft and fmm) in (a b c)
ddi_n_periodic_images    = [4, 4, 4]

### DDI cutoff radius (if cutoff is used)
ddi_radius               = 0.0

ddi_pb_zero_padding      = 1.0
```

*Anisotropy:*
By specifying a number of anisotropy axes via `n_anisotropy`, one
or more anisotropy axes can be set for the atoms in the basis cell. Specify columns
via headers: an index `i` and an axis `Kx Ky Kz` or `Ka Kb Kc`, as well as optionally
a magnitude `K`.

*Dipole-Dipole Interaction:*
Via the keyword `ddi_method` the method employed to calculate the dipole-dipole interactions is specified.

      `none`   -  Dipole-Dipole interactions are neglected
      `fft`    -  Uses a fast convolution method to accelerate the calculation (RECOMMENDED)
      `cutoff` -  Lets only spins within a maximal distance of 'ddi_radius' interact
      `fmm`    -  Uses the Fast-Multipole-Method (NOT YET IMPLEMENTED!)

If the `cutoff`-method has been chosen the cutoff-radius can be specified via `ddi_radius`.
*Note:* If `ddi_radius` < 0 a direct summation (i.e. brute force) over the whole system is performed. This is very inefficient and only encouraged for very small systems and/or unit-testing/debugging.

If the boundary conditions are periodic `ddi_n_periodic_images` specifies how many images are taken in the respective direction.
*Note:* The images are appended on both sides (the edges get filled too)
i.e. 1 0 0 → one image in +a direction and one image in -a direction

If the boundary conditions are open in a lattice direction and sufficiently many periodic images are chosen, zero-padding in that direction can be skipped.
This improves the speed and memory footprint of the calculation, but comes at the cost of a very slight asymmetry in the interactions (decreasing with increasing periodic images).
If `ddi_pb_zero_padding` is set to 1, zero-padding is performed - even if the boundary condition is periodic in a direction. If it is set to 0, zero-padding is skipped.

**Neighbour shells:**

Using `hamiltonian heisenberg_neighbours`, pair-wise interactions are handled in terms of
(isotropic) neighbour shells:

```toml
### Exchange: number of shells and constants [meV / unique pair]
Jij = [10.0, 1.0]

### DMI: number of shells and constants [meV / unique pair]
Dij = [6.0, 0.5]
### Chirality of DM vectors (+/-1=bloch, +/-2=neel)
dmi_chirality = 2
```

Note that pair-wise interaction parameters always mean energy per unique pair \<ij\>
(i.e. not per neighbour).

**Specify Pairs:**

You may alternatively input pair interaction explicitly as a table, giving you more
granular control over the system and the ability to specify non-isotropic interactions.

```toml
### Pairs
pairs = """
i j   da db dc    Jij   Dij  Dijx Dijy Dijz
0 0    1  0  0   10.0   6.0   1.0  0.0  0.0
0 0    0  1  0   10.0   6.0   0.0  1.0  0.0
0 0    0  0  1   10.0   6.0   0.0  0.0  1.0
"""

### Quadruplets
quadruplets = """
i    j  da_j  db_j  dc_j    k  da_k  db_k  dc_k    l  da_l  db_l  dc_l    Q
0    0  1     0     0       0  0     1     0       0  0     0     1       3.0
"""
```

Note that pair-wise interaction parameters always mean energy per unique pair \<ij\>
(not per neighbour).

*Pairs:*
Leaving out either exchange or DMI in the pairs is allowed and columns can
be placed in arbitrary order.
Note that instead of specifying the DM-vector as `Dijx Dijy Dijz`, you may specify it as
`Dija Dijb Dijc` if you prefer. You may also specify the magnitude separately as a column
`Dij`, but note that if you do, the vector (e.g. `Dijx Dijy Dijz`) will be normalized.
If the `Jij` or `Dij` keywords used for shells are present the associated columns in this
table will be ignored. This allows combining neighbour shell interactions with an
anisotropic part.

*Quadruplets:* Columns for these may also be placed in arbitrary order.

*Separate files:*
The anisotropy, pairs, and quadruplets can be placed into separate files.
In the configuration this is indicated by specifying the path prefixed by `file://`
instead of the table for any keyword:

In these files the table headers should to be at the top of the file, or you have to
specify the length of the table with the `n_pairs`, `n_quadruplets` or `n_anisotropy` keyword.

```toml
pairs = 'file://pairs.txt'
quadruplets = 'file://quadruplets.txt'
```

Note that the quadruplet interaction is defined as

![](https://math.vercel.app/?bgcolor=auto&from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright))

**Units:**

The external field is specified in Tesla, while anisotropy is specified in meV.
Pairwise interactions are specified in meV per unique pair `<ij>`,
while quadruplets are specified in meV per unique quadruplet `<ijkl>`.


Method Configuration
-------------------------------------------------------------------

### Method Output

For `llg` and equivalently `mc` and `gneb`, you can specify which
output you want your simulations to create.
These are specified in the `output` table under each method table.
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
