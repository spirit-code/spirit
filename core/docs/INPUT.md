SPIRIT INPUT FILES
====================

The following sections will list and explain the input file keywords.

General Settings and Log
--------------------

```Python
### Add a time tag to output files
output_tag_time         1
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


Geometry
--------------------

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


Hamiltonian
--------------------

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

### DM constant [meV]
n_neigh_shells_dmi        1
dij			  6.0

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
interaction_pairs_file        input/pairs.txt

### Quadruplets
interaction_quadruplets_file  input/quadruplets.txt
```

The input file for pairs should contain the following columns (in arbitrary order),
where leaving out either exchange or DMI is allowed:

```
i  j    da  db  dc    Dx  Dy  Dz    J
```

The input file for quadruplets needs to contain the following columns (in arbitrary order):

```
i    j  da_j  db_j  dc_j    k  da_k  db_k  dc_k    l  da_l  db_l  dc_l    Q
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