SPIRIT INPUT FILES
====================

The following sections will list and explain the input file keywords.

1. [General Settings and Log](#General)
2. [Geometry](#Geometry)
2. [Hamiltonian](#Hamiltonian)
2. [Method Output](#MethodOutput)
2. [Method Parameters](#MethodParameters)
2. [Pinning](#Pinning)
2. [Disorder and Defects](#Defects)


General Settings and Log <a name="General"></a>
--------------------------------------------------

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
--------------------------------------------------

The lattice constant scales everything you specify in basis and translations.

```Python
lattice_constant 1.0
```

The basis is apecified as three basis vectors `a`, `b` and `c`, together
with the number of atoms in the basis and their positions in terms of
the basis vectors.

```Python
#### a.x a.y a.z
#### b.x b.y b.z
#### c.x c.y c.z
#### n		     No of spins in the basic domain
#### 1.x 1.y 1.z     position of spins within basic
#### 2.x 2.y 2.z     domain in terms of basis vectors
basis
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1
0 0 0
```

The translations of the basis are specified as three vectors `(a,b,c)`, i.e.
they are given in terms of the basis vectors.

```Python
### Keyword translation_vectors ###
###   t1.a t1.b t1.c nCells(t1)
###   t2.a t2.b t2.c nCells(t2)
###   t3.a t3.b t3.c nCells(t3)
translation_vectors
1 0 0 100
0 1 0 100
0 0 1 1
```


Hamiltonian <a name="Hamiltonian"></a>
--------------------------------------------------

Note that you select the Hamiltonian you use with the `hamiltonian` keyword.

**Isotropic Heisenberg Hamiltonian**:

Interactions are handled in terms of neighbours.
You may specify shell-wise interaction parameters:

```Python
### Hamiltonian Type (heisenberg_neighbours, heisenberg_pairs, gaussian)
hamiltonian                heisenberg_neighbours

### boundary_conditions (in a b c) = 0(open), 1(periodical)
boundary_conditions        1 1 0

### external magnetic field vector[T]
external_field_magnitude   25.0
external_field_normal      0.0 0.0 1.0
### µSpin
mu_s                       2.0

### Uniaxial anisotropy constant [meV]
anisotropy_magnitude       0.0
anisotropy_normal          0.0 0.0 1.0

### Exchange constants [meV] for the respective shells
### Jij should appear after the >Number_of_neighbour_shells<
n_neigh_shells_exchange   2
jij 			  10.0  1.0

### Chirality of DM vectors (+/-1=bloch, +/-2=neel)
dm_chirality       2
### DM constant [meV]
n_neigh_shells_dmi 1
dij			       6.0

### Dipole-Dipole radius
dd_radius		  0.0
```

**Pair-wise Heisenberg Hamiltonian**:

Interactions are specified pair-wise. Single-threaded applications can thus
calculate interactions twice as fast as for the neighbour-wise case.
You may specify shell-wise interaction parameters.

```Python
### Hamiltonian Type (heisenberg_neighbours, heisenberg_pairs, gaussian)
hamiltonian                   heisenberg_pairs

### boundary_conditions (in a b c) = 0(open), 1(periodical)
boundary_conditions           1 1 0

### external magnetic field vector[T]
external_field_magnitude      25.0
external_field_normal         0.0 0.0 1.0
### µSpin
mu_s                          2.0

### Uniaxial anisotropy constant [meV]
anisotropy_magnitude          0.0
anisotropy_normal             0.0 0.0 1.0

### Dipole-Dipole radius
dd_radius                     0.0

### Pairs
n_interaction_pairs 2
i  j    da  db  dc    Dx  Dy  Dz     J
0  0    1   0   0     6.0 0.0 0.0    10.0
0  0    0   1   0     0.0 6.0 0.0    10.0

### Quadruplets
n_interaction_quadruplets 1
i    j  da_j  db_j  dc_j    k  da_k  db_k  dc_k    l  da_l  db_l  dc_l    Q
0    0  1     0     0       0  0     1     0       0  0     0     1       3.0
```

*Pairs*: Leaving out either exchange or DMI in the pairs is allowed and columns can
be placed in arbitrary order.
Note that instead of specifying the DM-vector as `Dx Dy Dz`, you may specify it as
`Da Db Dc` if you prefer, you may also specify the magnitude separately as a column `Dij`.

*Quadruplets*: Columns for these may also be placed in arbitrary order.

*Separate files*: The pairs and quadruplets can be placed into separate files.
If the pairs or quadruplets are at the top of the respective file, it is not necessary to specify `n_interaction_pairs` or `n_interaction_quadruplets` respectively.

```Python
### Pairs
interaction_pairs_file        input/pairs.txt

### Quadruplets
interaction_quadruplets_file  input/quadruplets.txt
```

**Gaussian Hamiltonian**:

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
--------------------------------------------------
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

**LLG**:
```Python
llg_output_energy_step             0    # Save system energy at each step
llg_output_energy_archive          1    # Archive system energy at each step
llg_output_energy_spin_resolved    0    # Also save energies for each spin
llg_output_energy_divide_by_nspins 1    # Normalize energies with number of spins

llg_output_configuration_step      1    # Save spin configuration at each step
llg_output_configuration_archive   0    # Archive spin configuration at each step
```

**MC**:
```Python
mc_output_energy_step             0
mc_output_energy_archive          1
mc_output_energy_spin_resolved    0
mc_output_energy_divide_by_nspins 1

mc_output_configuration_step    1
mc_output_configuration_archive 0
```

**GNEB**:
```Python
gneb_output_energies_step             0 # Save energies of images in chain
gneb_output_energies_interpolated     1 # Also save interpolated energies
gneb_output_energies_divide_by_nspins 1 # Normalize energies with number of spins

gneb_output_chain_step 0    # Save the whole chain at each step
```


Method Parameters <a name="MethodParameters"></a>
--------------------------------------------------
Again, the different Methods share a few common parameters.
On the example of the LLG Method:

```Python
### Maximum wall time for single simulation
### hh:mm:ss, where 0:0:0 is infinity
llg_max_walltime        0:0:0

### Force convergence parameter
llg_force_convergence   10e-9

### Number of iterations
llg_n_iterations		2000000
### Number of iterations after which to save
llg_n_iterations_log    2000
```

**LLG**:
```Python
### Seed for Random Number Generator
llg_seed			    20006

### Damping [none]
llg_damping				0.3E+0

### Time step dt
llg_dt					1.0E-3

### Temperature [K]
llg_temperature			0

### Spin transfer torque parameter proportional to injected current density
llg_stt_magnitude           0.0
### Spin current polarisation normal vector
llg_stt_polarisation_normal	1.0 0.0 0.0
```

**MC**:
```Python
### Seed for Random Number Generator
mc_seed			    20006

### Temperature [K]
mc_temperature	    0

### Acceptance ratio
mc_acceptance_ratio 0.5
```

**GNEB**:
```Python
### Constant for the spring force
gneb_spring_constant     1.0

### Number of energy interpolations between images
gneb_n_energy_interpolations 10
```


Pinning <a name="Pinning"></a>
--------------------------------------------------
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
### Specify the number of pinned sites and then the directions
### ispin S_x S_y S_z
n_pinned 3
0 1.0 0.0 0.0
1 0.0 1.0 0.0
2 0.0 0.0 1.0
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
--------------------------------------------------
Note that for this feature you need to build with `SPIRIT_ENABLE_DEFECTS`
set to `ON` in cmake.

Disorder is not yet implemented.
<!--Disorder is not yet implemented but you will specify the basis in the form
```Python
disorder 1
0  0.5
1  0.25
2  0.25
```-->

To specify defects, be it vacancies or impurities, you may fix atom types for
sites of the whole lattice by inserting a list into your input. For example:
```Python
### Atom types: type index 0..n or or vacancy (type < 0)
### Specify the number of defects and then the defects
### ispin itype
n_defects 3
0 -1
1 -1
2 -1
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