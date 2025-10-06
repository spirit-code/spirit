Hamiltonian
====================================================

*[configuration file](/core/docs/Input.md) section:* `[hamiltonian]`

![Hamiltonian](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20%5Csum_%5Clambda%20%5Cmathcal%7BH%7D_%5Clambda%5B%5Cvec%7Bn%7D%5D.svg)

<!-- $$ \mathcal{H}[\vec{n}] = \sum_\lambda \mathcal{H}_\lambda[\vec{n}] $$ -->

The full Spin-Hamiltonian is the sum over all interactions.

---

## Zeeman

![Zeeman](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_Z%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum_i%20%5Cmu_i%20%5Cvec%7BB%7D%5Ccdot%5Cvec%7Bn%7D_i.svg)

<!-- $$ \mathcal{H}_Z[\vec{n}] = - \sum_i \mu_i \vec{B}\cdot\vec{n}_i $$ -->

::::{tab-set}

:::{tab-item} TOML
```toml
### External magnetic field [T]
external_field.magnitude = 25.0
external_field.direction = [0.0, 0.0, 1.0]
```
:::

:::{tab-item} Python
```{eval-rst}
.. autofunction:: spirit.hamiltonian.set_field
   :no-index:
.. autofunction:: spirit.hamiltonian.get_field
   :no-index:
```
:::

:::{tab-item} C
```{eval-rst}
.. doxygenfunction:: Hamiltonian_Set_Field
.. doxygenfunction:: Hamiltonian_Get_Field
```
:::

::::



## Anisotropy
The anisotropy term is implemented in terms of three components:

### Uniaxial Anisotropy

![Uniaxial Anisotropy](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BA_1%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum_j%20K_j%20%28%5Chat%7BK%7D_j%5Ccdot%5Cvec%7Bn%7D_j%29%5E2%20%3D%20-%20%5Csum_j%20K_j%20%5B%5Ccos%28%5Ctheta_j%29%5D%5E2.svg)

<!-- $$ \mathcal{H}_{A_1}[\vec{n}] = - \sum_j K_j (\hat{K}_j\cdot\vec{n}_j)^2 = - \sum_j K_j [\cos(\theta_j)]^2 $$ -->

### Cubic Anisotropy

![Cubic Anisotropy](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BA_c%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_j%20K_j%20%28%5B%5Cvec%7Bn%7D_j%5D_x%5E4%20%2B%20%5B%5Cvec%7Bn%7D_j%5D_y%5E4%20%2B%20%5B%5Cvec%7Bn%7D_j%5D_z%5E4%29.svg)

<!-- $$ \mathcal{H}_{A_c}[\vec{n}] = - \frac{1}{2} \sum_j K_j ([\vec{n}_j]_x^4 + [\vec{n}_j]_y^4 + [\vec{n}_j]_z^4) $$ -->

::::{tab-set}

:::{tab-item} TOML
```toml
### Uniaxial anisotropy constant [meV]
anisotropy.magnitude     = 0.0
anisotropy.normal        = [0.0, 0.0, 1.0]

### Cubic anisotropy constant [meV]
cubic_anisotropy = 0.0
```

By specifying a number of anisotropy axes via `n_anisotropy`,
This way one or more anisotropy axes can be set for the atoms in the basis cell.
Specify columns via headers: an index `i` and an axis `Kx Ky Kz` or `Ka Kb Kc`,
as well as optionally a magnitude `K`. The optional column `K4` can be used to also 
specify the cubic anisotropy on a per-atom basis.
```toml
anisotropy = """
i  Kx Ky Kz    K   K4
0   0  0  1  1.0  0.0
"""
```
:::

:::{tab-item} Python
```{eval-rst}
.. autofunction:: spirit.hamiltonian.get_anisotropy
   :no-index:
.. autofunction:: spirit.hamiltonian.set_anisotropy
   :no-index:
```
:::

:::{tab-item} C
```{eval-rst}
.. doxygenfunction:: Hamiltonian_Set_Anisotropy
.. doxygenfunction:: Hamiltonian_Get_Anisotropy
```
:::

::::



## Biaxial Anisotropy

![Biaxial Anisotropy](https://math.vercel.app/?bgcolor=auto&from=%5Cbegin%7Balignedat%7D%7B1%7D%0A%5Cmathcal%7BH%7D_%7BA_2%7D%20%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%281%20-%20%28%5Chat%7BK%7D_j%5E%7B%281%29%7D%5Ccdot%5Cvec%7Bn%7D_j%29%5E2%29%5E%7Bn_1%7D%20%5Ccdot%20%28%5Chat%7BK%7D_j%5E%7B%282%29%7D%5Ccdot%5Cvec%7Bn%7D_j%29%5E%7Bn_2%7D%20%5Ccdot%20%28%28%5Chat%7BK%7D_j%5E%7B%281%29%7D%20%5Ctimes%20%5Chat%7BK%7D_j%5E%7B%282%29%7D%20%29%20%5Ccdot%5Cvec%7Bn%7D_j%29%5E%7Bn_3%7D%20%5C%5C%0A%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%5Ccdot%20%5B%5Csin%28%5Ctheta_j%29%5D%5E%7B2n_1%7D%20%5Ccdot%20%5B%5Ccos%28%5Cvarphi_j%29%5Csin%28%5Ctheta_j%29%5D%5E%7Bn_2%7D%20%5Ccdot%20%5B%5Csin%28%5Cvarphi_j%29%5Csin%28%5Ctheta_j%29%5D%5E%7Bn_3%7D%20%5C%5C%0A%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%5Ccdot%20%5B%5Csin%28%5Ctheta_j%29%5D%5E%7B2n_1%20%2B%20n_2%20%2B%20n_3%7D%20%5Ccdot%20%5B%5Ccos%28%5Cvarphi_j%29%5D%5E%7Bn_2%7D%20%5Ccdot%20%5B%5Csin%28%5Cvarphi_j%29%5D%5E%7Bn_3%7D%20%5C%5C%0A%5Cend%7Balignedat%7D.svg)

<!-- $$
\begin{alignedat}{1}
\mathcal{H}_{A_2} =& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} (1 - (\hat{K}_j^{(1)}\cdot\vec{n}_j)^2)^{n_1} \cdot (\hat{K}_j^{(2)}\cdot\vec{n}_j)^{n_2} \cdot ((\hat{K}_j^{(1)} \times \hat{K}_j^{(2)} ) \cdot\vec{n}_j)^{n_3} \\
=& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} \cdot [\sin(\theta_j)]^{2n_1} \cdot [\cos(\varphi_j)\sin(\theta_j)]^{n_2} \cdot [\sin(\varphi_j)\sin(\theta_j)]^{n_3} \\
=& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} \cdot [\sin(\theta_j)]^{2n_1 + n_2 + n_3} \cdot [\cos(\varphi_j)]^{n_2} \cdot [\sin(\varphi_j)]^{n_3} \\
\end{alignedat}
$$ -->

Where for any site $j$ the vectors $\hat{K}_j^{(1)}$, $\hat{K}_j^{(2)}$ and $\hat{K}_j^{(3)} = \hat{K}_j^{(1)} \times \hat{K}_j^{(2)}$ are pairwise orthonormal.


The uniaxial anisotropy is equivalent to the biaxial anisotropy up to an offset to the total energy when setting

![equivalence](https://math.vercel.app/?bgcolor=auto&from=%5Cbegin%7Balignedat%7D%7B1%7D%0A%5Chat%7BK%7D_j%5E%7B%281%29%7D%20%3D%26%20%5Chat%7BK%7D_j%20%5C%5C%0AK_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%3D%26%20-K_j%20%5Cdelta_%7Bn_1%2C%201%7D%5Cdelta_%7Bn_2%2C%200%7D%5Cdelta_%7Bn_3%2C%200%7D.%0A%5Cend%7Balignedat%7D%0A.svg)

<!-- $$
\begin{alignedat}{1}
\hat{K}_j^{(1)} =& \hat{K}_j \\
K_j^{(n_1, n_2, n_3)} =& -K_j \delta_{n_1, 1}\delta_{n_2, 0}\delta_{n_3, 0}.
\end{alignedat}
$$ -->

::::{tab-set}

:::{tab-item} TOML

The configuration uses two [tables](/core/docs/Input.md#tables): `biaxial_anisotropy_axes` and `biaxial_anisotropy_terms`.
```toml
### Biaxial anisotropy axes (normalized)
biaxial_anisotropy_axes = """
i  k1x k1y k1z  k2x k2y k2z
0  0.0 0.0 1.0  1.0 0.0 0.0
"""
### Biaxial anisotropy terms with k [meV]
### This example is equivalent to a uniaxial anisotropy
biaxial_anisotropy_terms = """
i  n1 n2 n3    K
0   1  0  0  1.0
"""
```

:::

:::{tab-item} Python
```{eval-rst}
.. autofunction:: spirit.hamiltonian.get_biaxial_anisotropy
   :no-index:
.. autofunction:: spirit.hamiltonian.set_biaxial_anisotropy
   :no-index:
```
:::

:::{tab-item} C
```{eval-rst}
.. doxygenfunction:: Hamiltonian_Set_Biaxial_Anisotropy
.. doxygenfunction:: Hamiltonian_Get_Biaxial_Anisotropy
```
:::

::::



## Pair interactions

### Heisenberg exchange

![Exchange](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BXC%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20J_%7Bij%7D%20%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j.svg)

<!-- $$ \mathcal{H}_{XC}[\vec{n}] = - \sum\limits_{\braket{ij}}\, J_{ij} \vec{n}_i\cdot\vec{n}_j $$ -->

### Dzyaloshinskii–Moriya Interaction (DMI)

![DMI](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BDMI%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20%5Cvec%7BD%7D_%7Bij%7D%20%5Ccdot%20%28%5Cvec%7Bn%7D_i%5Ctimes%5Cvec%7Bn%7D_j%29%0A.svg)

<!-- $$ \mathcal{H}_{DMI}[\vec{n}] = - \sum\limits_{\braket{ij}}\, \vec{D}_{ij} \cdot (\vec{n}_i\times\vec{n}_j) $$ -->

where it is important to note that `<ij>` denotes the unique pairs of interacting spins `i` and `j`.
Consequently, pair-wise interaction parameters always mean energy per unique pair \<ij\>
(i.e., not per neighbour).

::::{tab-set}

:::{tab-item} TOML
Pair-wise interactions are typically handled in terms of (isotropic) neighbour shells:

```toml
### Exchange: number of shells and constants [meV / unique pair]
Jij = [10.0, 1.0]

### DMI: number of shells and constants [meV / unique pair]
Dij = [6.0, 0.5]
### Chirality of DM vectors (+/-1=bloch, +/-2=neel)
dmi_chirality = 2
```
You may alternatively input pair interaction explicitly as a [table](/core/docs/Input.md#tables), giving you more
granular control over the system and the ability to specify non-isotropic interactions.

```toml
### Pairs
pairs = """
i j   da db dc    Jij   Dij  Dijx Dijy Dijz
0 0    1  0  0   10.0   6.0   1.0  0.0  0.0
0 0    0  1  0   10.0   6.0   0.0  1.0  0.0
0 0    0  0  1   10.0   6.0   0.0  0.0  1.0
"""

```

Leaving out either exchange or DMI in the pairs is allowed and columns can be placed in arbitrary order.
Note that instead of specifying the DM-vector as `Dijx Dijy Dijz`, you may specify it as `Dija Dijb Dijc` if you prefer.
You may also specify the magnitude separately as a column `Dij`, but note that if you do, the vector (e.g. `Dijx Dijy Dijz`) will be normalized.
If the `Jij` or `Dij` keywords used for shells are present the associated columns in this table will be ignored. This allows combining neighbour shell interactions with an anisotropic part.

:::

:::{tab-item} Python
```{eval-rst}
.. autodata:: spirit.hamiltonian.CHIRALITY_BLOCH
   :no-index:
.. autodata:: spirit.hamiltonian.CHIRALITY_NEEL
   :no-index:
.. autodata:: spirit.hamiltonian.CHIRALITY_BLOCH_INVERSE
   :no-index:
.. autodata:: spirit.hamiltonian.CHIRALITY_NEEL_INVERSE
   :no-index:
.. autofunction:: spirit.hamiltonian.set_dmi
   :no-index:
.. autofunction:: spirit.hamiltonian.set_exchange
   :no-index:
```
:::

:::{tab-item} C
```{eval-rst}
.. doxygendefine:: SPIRIT_CHIRALITY_BLOCH
.. doxygendefine:: SPIRIT_CHIRALITY_NEEL
.. doxygendefine:: SPIRIT_CHIRALITY_BLOCH_INVERSE
.. doxygendefine:: SPIRIT_CHIRALITY_NEEL_INVERSE

.. doxygenfunction:: Hamiltonian_Set_DMI
.. doxygenfunction:: Hamiltonian_Get_DMI_Shells
.. doxygenfunction:: Hamiltonian_Get_DMI_N_Pairs

.. doxygenfunction:: Hamiltonian_Set_Exchange
.. doxygenfunction:: Hamiltonian_Get_Exchange_Shells
.. doxygenfunction:: Hamiltonian_Get_Exchange_N_Pairs
.. doxygenfunction:: Hamiltonian_Get_Exchange_Pairs
```
:::

::::


## Two-Site Anisotropy

![Two-Site Anisotropy](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BA%5E2%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%5Csum_%7B%5Cbraket%7Bij%7D%7D%20%5Cvec%7Bn%7D_i%20%5Ccdot%20K_%7Bij%7D%20%5Cvec%7Bn%7D_i.svg)

<!-- \mathcal{H}_{A^2}[\vec{n}] = -\sum_{\braket{ij}} \vec{n}_i \cdot K_{ij} \vec{n}_i -->


The Two-Site Anisotropy interaction covers the symmetric, traceless part of the two-site interaction tensor.
It is specified in terms of the symmetry reduced Cartesian coordinates in a [table](/core/docs/Input.md#tables).
If one of the diagonal entries is omitted it will be computed from the other two to ensure that the trace of the tensor vanishes.

:::{warning}
This interaction can overlap with the [Heisenberg exchange](#heisenberg-exchange), if $K_{ij}^{xx} + K_{ij}^{yy} + K_{ij}^{zz} \neq 0$.
:::

::::{tab-set}

:::{tab-item} TOML
```toml
### Two Site Anisotropy in meV per pair
two_site_anisotropy = """
i j   da db dc   Kijxx  Kijyy  Kijzz  Kijyz  Kijxz  Kijxy
0 0    1  0  0    -1.0    0.5    0.5    0.0    0.0    0.0
0 0    0  1  0     0.5   -1.0    0.5    0.0    0.0    0.0
0 0    0  0  1     0.5    0.5   -1.0    0.0    0.0    0.0
"""
```
:::

:::{tab-item} Python
Not yet implemented
:::

:::{tab-item} C
Not yet implemented
:::

::::



## Dipole-Dipole Interaction

![Dipole-Dipole Interaction](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D_%7BDDI%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cmu_0%7D%7B4%5Cpi%7D%20%5Csum_%7B%5Csubstack%7Bi%2Cj%20%5C%5C%20i%20%5Cneq%20j%7D%7D%20%5Cmu_i%20%5Cmu_j%20%5Cfrac%7B%28%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Chat%7Br%7D_%7Bij%7D%29%20%28%5Cvec%7Bn%7D_j%5Ccdot%5Chat%7Br%7D_%7Bij%7D%29%20-%20%5Cvec%7Bn%7D_i%20%5Cvec%7Bn%7D_j%7D%7B%7Br_%7Bij%7D%7D%5E3%7D%0A.svg)

<!-- $$
\mathcal{H}_{DDI}[\vec{n}]
    = \frac{1}{2}\frac{\mu_0}{4\pi} \sum_{\substack{i,j \\ i \neq j}} \mu_i \mu_j \frac{(\vec{n}_i \cdot \hat{r}_{ij}) (\vec{n}_j\cdot\hat{r}_{ij}) - \vec{n}_i \vec{n}_j}{{r_{ij}}^3}
$$ -->

Via the keyword `ddi_method` the method employed to calculate the dipole-dipole interactions is specified.

      `none`   -  Dipole-Dipole interactions are neglected
      `fft`    -  Uses a fast convolution method to accelerate the calculation (RECOMMENDED)
      `cutoff` -  Lets only spins within a maximal distance of 'ddi_radius' interact
      `fmm`    -  Uses the Fast-Multipole-Method (NOT YET IMPLEMENTED!)

If the `cutoff`-method has been chosen the cutoff-radius can be specified via `ddi_radius`.
*Note:* If `ddi_radius` < 0 a direct summation (i.e. brute force) over the whole system is performed. This is very inefficient and only encouraged for very small systems and/or unit-testing/debugging.

If the boundary conditions are periodic `ddi_n_periodic_images` specifies how many images are taken in the respective direction.
*Note:* The images are appended on both sides (the edges get filled too) i.e. 1 0 0 → one image in +a direction and one image in -a direction

If the boundary conditions are open in a lattice direction and sufficiently many periodic images are chosen, zero-padding in that direction can be skipped.
This improves the speed and memory footprint of the calculation, but comes at the cost of a very slight asymmetry in the interactions (decreasing with increasing periodic images).
If `ddi_pb_zero_padding` is set to 1, zero-padding is performed - even if the boundary condition is periodic in a direction. If it is set to 0, zero-padding is skipped.

::::{tab-set}

:::{tab-item} TOML
```toml
### Dipole-dipole interaction caclulation method
### (none, fft, fmm, cutoff)
ddi_method               = 'fft'

### DDI number of periodic images (fft and fmm) in (a b c)
ddi_n_periodic_images    = [4, 4, 4]

### DDI cutoff radius (if cutoff is used)
ddi_radius               = 0.0

ddi_pb_zero_padding      = 1.0
```
:::

:::{tab-item} Python
```{eval-rst}
.. autodata:: spirit.hamiltonian.DDI_METHOD_NONE
   :no-index:
.. autodata:: spirit.hamiltonian.DDI_METHOD_FFT
   :no-index:
.. autodata:: spirit.hamiltonian.DDI_METHOD_CUTOFF
   :no-index:
.. autofunction:: spirit.hamiltonian.get_ddi
   :no-index:
.. autofunction:: spirit.hamiltonian.set_ddi
   :no-index:
```
:::

:::{tab-item} C
```{eval-rst}
.. doxygendefine:: SPIRIT_DDI_METHOD_NONE
.. doxygendefine:: SPIRIT_DDI_METHOD_FFT
.. doxygendefine:: SPIRIT_DDI_METHOD_CUTOFF
.. doxygenfunction:: Hamiltonian_Set_DDI
.. doxygenfunction:: Hamiltonian_Get_DDI
```
:::

::::



## Quadruplet Interaction

![Quadruplet Interaction](https://math.vercel.app/?bgcolor=auto&from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright))

<!-- $$ \mathcal{H}_{Quad}[\vec{n}] = - \sum_{i,j,k,l} K_{ijkl} (\vec{n}_i \cdot \vec{n}_j)(\vec{n}_k \cdot \vec{n}_l) $$ -->

Quadruplets are specified in meV per unique quadruplet `<ijkl>`.

::::{tab-set}

:::{tab-item} TOML
```toml
### Quadruplets in meV per unique quadruplet <ijkl>
quadruplets = """
i    j  da_j  db_j  dc_j    k  da_k  db_k  dc_k    l  da_l  db_l  dc_l    Q
0    0  1     0     0       0  0     1     0       0  0     0     1       3.0
"""
```

Columns for these may also be placed in arbitrary order.
:::

:::{tab-item} Python
Not yet implemented
:::

:::{tab-item} C
Not yet implemented
:::

::::

## Custom Interaction

The Hamiltonian implementation is modular, so custom interactions can be easily implemented.
See [**Custom Interaction**](/core/docs/Custom_Interaction.md) for details.
