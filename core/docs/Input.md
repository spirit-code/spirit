Spirit inputfile
====================================================

The following sections will list and explain the input file keywords.

1. [General Settings and Log](#General)
2. [Geometry](#Geometry)
3. [Heisenberg Hamiltonian](#Heisenberg)
4. [Gaussian Hamiltonian](#Gaussian)
5. [Method Output](#MethodOutput)
6. [Method Parameters](#MethodParameters)
7. [Pinning](#Pinning)
8. [Disorder and Defects](#Defects)


General Settings and Log <a name="General"></a>
----------------------------------------------------

```Python
### Add a tag to output files (for timestamp use "<time>")
output_file_tag         some_tag
```

```Python
### Save input parameters on creation of State
log_input_save_initial  0
### Save input parameters on deletion of State
log_input_save_final    0

### Print log messages to the console
log_to_console    1
### Print messages up to (including) log_console_level
log_console_level 5

### Save the log as a file
log_to_file    1
### Save messages up to (including) log_file_level
log_file_level 5
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


Geometry <a name="Geometry"></a>
----------------------------------------------------

The Geometry of a spin system is specified in form of a bravais lattice
and a basis cell of atoms. The number of basis cells along each principal
direction of the basis can be specified.
*Note:* the default basis is a single atom at (0,0,0).

**3D simple cubic example:**

```Python
### The bravais lattice type
bravais_lattice sc

### µSpin
mu_s 2.0

### Number of basis cells along principal
### directions (a b c)
n_basis_cells 100 100 10
```

If you have a nontrivial basis cell, note that you should specify `mu_s`
for all atoms in your basis cell (see the next example).

**2D honeycomb example:**

```Python
### The bravais lattice type
bravais_lattice hex2d

### The basis cell in units of bravais vectors
### n            No of spins in the basis cell
### 1.x 1.y 1.z  position of spins within basis
### 2.x 2.y 2.z  cell in terms of bravais vectors
basis
2
0          0         0
0.33333333 0.3333333 0

### µSpin
mu_s 2.0 1.0

### Number of basis cells along principal
### directions (a b c)
n_basis_cells 100 100 1
```

The bravais lattice can be one of the following:

| Bravais Lattice Type     | Keyword  | Comment                     |
| ------------------------ | -------- | --------------------------- |
| Simple cubic             | sc       |                             |
| Body-centered cubic      | bcc      |                             |
| Face-centered cubic      | fcc      |                             |
| Hexagonal (2D)           | hex2d    |  60deg angle                |
| Hexagonal (2D)           | hex2d60  |  60deg angle                |
| Hexagonal (2D)           | hex2d120 | 120deg angle                |
| Hexagonal closely packed | hcp      | 120deg, not yet implemented |
| Hexagonal densely packed | hdp      |  60deg, not yet implemented |
| Rhombohedral             | rho      | not yet implemented         |
| Simple-tetragonal        | stet     | not yet implemented         |
| Simple-orthorhombic      | so       | not yet implemented         |
| Simple-monoclinic        | sm       | not yet implemented         |
| Simple triclinic         | stri     | not yet implemented         |

Alternatively it can be input manually, either through vectors
or as the bravais matrix:

```Python
### bravais_vectors or bravais_matrix
###   a.x a.y a.z       a.x b.x c.x
###   b.x b.y b.z       a.y b.y c.y
###   c.x c.y c.z       a.z b.z c.z
bravais_vectors
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
```

A lattice constant can be used for scaling:
```Python
### Scaling constant
lattice_constant 1.0
```
Note that it scales the Bravais vectors and therefore the
translations, atom positions in the basis cell and potentially
-- if you specified them in terms of the Bravais vectors --
also the anisotropy and DM vectors.

**Units:**

The Bravais vectors (or matrix) are specified in Cartesian coordinates in units of Angstrom.
The basis atoms are specified in units of the Bravais vectors.

The atomic moments `mu_s` are specified in units of the Bohr magneton `mu_B`.


Heisenberg Hamiltonian <a name="Heisenberg"></a>
----------------------------------------------------

To use a Heisenberg Hamiltonian, use either `heisenberg_neighbours` or `heisenberg_pairs`
as input parameter after the `hamiltonian` keyword.

The Hamiltonian is defined as

![](https://math.vercel.app/?bgcolor=auto&from=%0A%09%5Cmathcal%7BH%7D%20%3D%0A%20%20%20%20%20%20-%20%5Csum_i%20%5Cmu_i%20%5Cvec%7BB%7D%5Ccdot%5Cvec%7Bn%7D_i%0A%20%20%20%20%20%20%20-%20%5Csum_i%20%5Csum_j%20K_j%20%28%5Chat%7BK%7D_j%5Ccdot%5Cvec%7Bn%7D_i%29%5E2%5C%5C%0A%20%20%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20J_%7Bij%7D%20%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%0A%20%20%20%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20%5Cvec%7BD%7D_%7Bij%7D%20%5Ccdot%20(%5Cvec%7Bn%7D_i%5Ctimes%5Cvec%7Bn%7D_j)%5C%5C%0A%20%20%20%20%20%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cmu_0%7D%7B4%5Cpi%7D%20%5Csum_%7B%5Csubstack%7Bi%2Cj%20%5C%5C%20i%20%5Cneq%20j%7D%7D%20%5Cmu_i%20%5Cmu_j%20%5Cfrac%7B(%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Chat%7Br%7D_%7Bij%7D)%20(%5Cvec%7Bn%7D_j%5Ccdot%5Chat%7Br%7D_%7Bij%7D)%20-%20%5Cvec%7Bn%7D_i%20%5Cvec%7Bn%7D_j%7D%7B%7Br_%7Bij%7D%7D%5E3%7D)

where `<ij>` denotes the unique pairs of interacting spins `i` and `j`.
For more details, such as the notation used here, see [Phys. Rev. B **99** 224414 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.224414).

**General Parameters:**

```Python
### Hamiltonian Type (heisenberg_neighbours, heisenberg_pairs, gaussian)
hamiltonian              heisenberg_neighbours

### Boundary conditions (in a b c) = 0(open), 1(periodical)
boundary_conditions      1 1 0

### External magnetic field [T]
external_field_magnitude 25.0
external_field_normal    0.0 0.0 1.0

### Uniaxial anisotropy constant [meV]
anisotropy_magnitude     0.0
anisotropy_normal        0.0 0.0 1.0

### Dipole-dipole interaction caclulation method
### (none, fft, fmm, cutoff)
ddi_method               fft

### DDI number of periodic images (fft and fmm) in (a b c)
ddi_n_periodic_images    4 4 4

### DDI cutoff radius (if cutoff is used)
ddi_radius               0.0

ddi_pb_zero_padding      1.0
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
i.e. 1 0 0 -> one image in +a direction and one image in -a direction

If the boundary conditions are open in a lattice direction and sufficiently many periodic images are chosen, zero-padding in that direction can be skipped.
This improves the speed and memory footprint of the calculation, but comes at the cost of a very slight asymmetry in the interactions (decreasing with increasing periodic images).
If `ddi_pb_zero_padding` is set to 1, zero-padding is performed - even if the boundary condition is periodic in a direction. If it is set to 0, zero-padding is skipped.

**Neighbour shells:**

Using `hamiltonian heisenberg_neighbours`, pair-wise interactions are handled in terms of
(isotropic) neighbour shells:

```Python
### Hamiltonian Type (heisenberg_neighbours, heisenberg_pairs, gaussian)
hamiltonian       heisenberg_neighbours

### Exchange: number of shells and constants [meV / unique pair]
n_shells_exchange 2
jij               10.0  1.0

### Chirality of DM vectors (+/-1=bloch, +/-2=neel)
dm_chirality      2
### DMI: number of shells and constants [meV / unique pair]
n_shells_dmi      2
dij	              6.0 0.5
```

Note that pair-wise interaction parameters always mean energy per unique pair \<ij\>
(i.e. not per neighbour).

**Specify Pairs:**

Using `hamiltonian heisenberg_pairs`, you may input interactions explicitly,
in form of unique pairs \<ij\>, giving you more granular control over the system and
the ability to specify non-isotropic interactions:

```Python
### Hamiltonian Type (heisenberg_neighbours, heisenberg_pairs, gaussian)
hamiltonian       heisenberg_pairs

### Pairs
n_interaction_pairs 3
i j   da db dc    Jij   Dij  Dijx Dijy Dijz
0 0    1  0  0   10.0   6.0   1.0  0.0  0.0
0 0    0  1  0   10.0   6.0   0.0  1.0  0.0
0 0    0  0  1   10.0   6.0   0.0  0.0  1.0

### Quadruplets
n_interaction_quadruplets 1
i    j  da_j  db_j  dc_j    k  da_k  db_k  dc_k    l  da_l  db_l  dc_l    Q
0    0  1     0     0       0  0     1     0       0  0     0     1       3.0
```

Note that pair-wise interaction parameters always mean energy per unique pair \<ij\>
(not per neighbour).

*Pairs:*
Leaving out either exchange or DMI in the pairs is allowed and columns can
be placed in arbitrary order.
Note that instead of specifying the DM-vector as `Dijx Dijy Dijz`, you may specify it as
`Dija Dijb Dijc` if you prefer. You may also specify the magnitude separately as a column
`Dij`, but note that if you do, the vector (e.g. `Dijx Dijy Dijz`) will be normalized.

*Quadruplets:* Columns for these may also be placed in arbitrary order.

*Separate files:*
The anisotropy, pairs and quadruplets can be placed into separate files,
you can use `anisotropy_from_file`, `pairs_from_file` and `quadruplets_from_file`.

If the headers for anisotropies, pairs or quadruplets are at the top of the respective file,
it is not necessary to specify `n_anisotropy`, `n_interaction_pairs` or `n_interaction_quadruplets`
respectively.

```Python
### Pairs
interaction_pairs_file       input/pairs.txt

### Quadruplets
interaction_quadruplets_file input/quadruplets.txt
```

Note that the quadruplet interaction is defined as

![](https://math.vercel.app/?bgcolor=auto&from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright))

**Units:**

The external field is specified in Tesla, while anisotropy is specified in meV.
Pairwise interactions are specified in meV per unique pair \<ij\>,
while quadruplets are specified in meV per unique quadruplet \<ijkl\>.


Gaussian Hamiltonian <a name="Gaussian"></a>
----------------------------------------------------

Note that you select the Hamiltonian you use with the `hamiltonian gaussian` input option.

This is a testing Hamiltonian consisting of the superposition
of gaussian potentials. It does not contain interactions.

```Python
hamiltonian gaussian

### Number of Gaussians
n_gaussians 2

### Gaussians
###   a is the amplitude, s is the width, c the center
###   the directions c you enter will be normalized
###   a1 s1 c1.x c1.y c1.z
###   ...
gaussians
 1    0.2   -1   0   0
 0.5  0.4    0   0  -1
```


Method Output <a name="MethodOutput"></a>
----------------------------------------------------

For `llg` and equivalently `mc` and `gneb`, you can specify which
output you want your simulations to create. They share a few common
output types, for example:

```Python
llg_output_any     1    # Write any output at all
llg_output_initial 1    # Save before the first iteration
llg_output_final   1    # Save after the last iteration
```

Note in the following that `step` means after each `N` iterations and
denotes a separate file for each step, whereas `archive` denotes that
results are appended to an archive file at each step.

The energy output files are in units of meV, and can be switched to
meV per spin with `<method>_output_energy_divide_by_nspins`.

**LLG:**
```Python
llg_output_energy_step             0    # Save system energy at each step
llg_output_energy_archive          1    # Archive system energy at each step
llg_output_energy_spin_resolved    0    # Also save energies for each spin
llg_output_energy_divide_by_nspins 1    # Normalize energies with number of spins

llg_output_configuration_step      1    # Save spin configuration at each step
llg_output_configuration_archive   0    # Archive spin configuration at each step
```

**MC:**
```Python
mc_output_energy_step             0
mc_output_energy_archive          1
mc_output_energy_spin_resolved    0
mc_output_energy_divide_by_nspins 1

mc_output_configuration_step    1
mc_output_configuration_archive 0
```

**GNEB:**
```Python
gneb_output_energies_step             0 # Save energies of images in chain
gneb_output_energies_interpolated     1 # Also save interpolated energies
gneb_output_energies_divide_by_nspins 1 # Normalize energies with number of spins

gneb_output_chain_step 0    # Save the whole chain at each step
```


Method Parameters <a name="MethodParameters"></a>
----------------------------------------------------

Again, the different Methods share a few common parameters.
On the example of the LLG Method:

```Python
### Maximum wall time for single simulation
### hh:mm:ss, where 0:0:0 is infinity
llg_max_walltime        0:0:0

### Force convergence parameter
llg_force_convergence   10e-9

### Number of iterations
llg_n_iterations        2000000
### Number of iterations after which to save
llg_n_iterations_log    2000
### Number of iterations that gets run with no checks or outputs (Increasing this boosts performance, especially in CUDA builds)
llg_n_iterations_amortize 1

```

**LLG:**

```Python
### Seed for Random Number Generator
llg_seed            20006

### Damping [none]
llg_damping         0.3E+0

### Time step dt [ps]
llg_dt              1.0E-3

### Temperature [K]
llg_temperature	    0
llg_temperature_gradient_direction   1 0 0
llg_temperature_gradient_inclination 0.0

### Spin transfer torque parameter proportional to injected current density
llg_stt_magnitude   0.0
### Spin current polarisation normal vector
llg_stt_polarisation_normal	1.0 0.0 0.0
```

The time step `dt` is given in picoseconds.
The temperature is given in Kelvin and the temperature gradient in Kelvin/Angstrom.

If you don't specify a seed for the RNG, it will be chosen randomly.

**MC:**

```Python
### Seed for Random Number Generator
mc_seed	            20006

### Temperature [K]
mc_temperature      0

### Acceptance ratio
mc_acceptance_ratio 0.5
```

The temperature is given in Kelvin.

If you don't specify a seed for the RNG, it will be chosen randomly.

**GNEB:**

```Python
### Constant for the spring force
gneb_spring_constant 1.0

### Number of energy interpolations between images
gneb_n_energy_interpolations 10
```


Pinning <a name="Pinning"></a>
----------------------------------------------------

Note that for this feature you need to build with `SPIRIT_ENABLE_PINNING`
set to `ON` in cmake.

For each lattice direction `a` `b` and `c`, you have two choices for pinning.
For example to pin `n` cells in the `a` direction, you can set both
`pin_na_left` and `pin_na_right` to different values or set `pin_na` to set
both to the same value.
To set the direction of the pinned cells, you need to give the `pinning_cell`
keyword and one vector for each basis atom.

You can for example do the following to create a U-shaped pinning in x-direction:
```Python
# Pin left side of the sample (2 rows)
pin_na_left 2
# Pin top and bottom sides (2 rows each)
pin_nb      2
# Pin the atoms to x-direction
pinning_cell
1 0 0
```

To specify individual pinned sites (overriding the above pinning settings),
insert a list into your input. For example:
```Python
### Specify the number of pinned sites and then the sites (in terms of translations) and directions
### i  da db dc  Sx Sy Sz
n_pinned 3
0  0 0 0  1.0 0.0 0.0
0  1 0 0  0.0 1.0 0.0
0  0 1 0  0.0 0.0 1.0
```
You may also place it into a separate file with the keyword `pinned_from_file`,
e.g.
```Python
### Read pinned sites from a separate file
pinned_from_file input/pinned.txt
```
The file should either contain only the pinned sites or you need to specify `n_pinned`
inside the file.


Disorder and Defects <a name="Defects"></a>
----------------------------------------------------

Note that for this feature you need to build with `SPIRIT_ENABLE_DEFECTS`
set to `ON` in cmake.

In order to specify disorder across the lattice, you can write for example a
single atom basis with 50% chance of containing one of two atom types (0 or 1):
```Python
# iatom  atom_type  mu_s  concentration
atom_types 1
    0        1       2.0     0.5
```

Note that you have to also specify the magnetic moment, as this is now site-
and atom type dependent.

A two-atom basis where
- the first atom is type 0
- the second atom is 70% type 1 and 30% type 2
```Python
# iatom  atom_type  mu_s  concentration
atom_types 2
    0        0       1.0      1
    1        1       2.5     0.7
    1        2       2.3     0.3
```

The total concentration on a site should not be more than `1`. If it is less
than `1`, vacancies will appear.

To specify defects, be it vacancies or impurities, you may fix atom types for
sites of the whole lattice by inserting a list into your input. For example:
```Python
### Atom types: type index 0..n or or vacancy (type < 0)
### Specify the number of defects and then the defects in terms of translations and type
### i  da db dc  itype
n_defects 3
0  0 0 0  -1
0  1 0 0  -1
0  0 1 0  -1
```
You may also place it into a separate file with the keyword `defects_from_file`,
e.g.
```Python
### Read defects from a separate file
defects_from_file input/defects.txt
```
The file should either contain only the defects or you need to specify `n_defects`
inside the file.


---

[Home](Readme.md)