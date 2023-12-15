Heisenberg Hamiltonian
====================================================

### Hamiltonian
The Spin-Hamiltonian is defined as
$$
\mathcal{H}[\vec{n}] =
    \mathcal{H}_{Z}[\vec{n}]
    + \mathcal{H}_{A}[\vec{n}]
    + \mathcal{H}_{XC}[\vec{n}]
    + \mathcal{H}_{DMI}[\vec{n}]
    + \mathcal{H}_{DDI}[\vec{n}]
    + \mathcal{H}_{Quad}[\vec{n}]
$$

---

### Zeeman
$$ \mathcal{H}_Z[\vec{n}] = - \sum_i \mu_i \vec{B}\cdot\vec{n}_i $$

---

### Anisotropy
The anisotropy term is implemented in three variants:
- uniaxial anisotropy
$$ \mathcal{H}_{A_1}[\vec{n}] = - \sum_i \sum_j K_j (\hat{K}_j\cdot\vec{n}_i)^2 = - \sum_i \sum_j K_j [\cos(\theta_j)]^2 $$
- cubic anisotropy
$$ \mathcal{H}_{A_c}[\vec{n}] = - \frac{1}{2} \sum_i \sum_j K_j ([\vec{n}_i]_x^4 + [\vec{n}_i]_y^4 + [\vec{n}_i]_z^4) $$
- biaxial anisotropy
$$
\begin{alignedat}{1}
\mathcal{H}_{AC} =& \sum_{i,j} K_j^{(n_1, n_2, n_3)} (1 - (\hat{K}_j^{(1)}\cdot\vec{n}_i)^2)^{n_1} \cdot (\hat{K}_j^{(2)}\cdot\vec{n}_i)^{n_2} \cdot ((\hat{K}_j^{(1)} \times \hat{K}_j^{(2)} ) \cdot\vec{n}_i)^{n_3} \\
=& \sum_{i,j} K_j^{(n_1, n_2, n_3)} \cdot [\sin(\theta_j)]^{2n_1} \cdot [\cos(\varphi_j)\sin(\theta_j)]^{n_2} \cdot [\sin(\varphi_j)\sin(\theta_j)]^{n_3} \\
\end{alignedat}
$$

The uniaxial anisotropy is equivalent to the biaxial anisotropy up to an offset to the total energy when setting
$$
\begin{alignedat}{1}
\hat{K}_j^{(1)} =& \hat{K}_j \\
K_j^{(n_1, n_2, n_3)} =& -K_j \delta_{n_1, 1}\delta_{n_2, 0}\delta_{n_3, 0}.
\end{alignedat}
$$

---

### Exchange
- symmetric exchange interaction
$$ \mathcal{H}_{XC}[\vec{n}] = - \sum\limits_{\braket{ij}}\, J_{ij} \vec{n}_i\cdot\vec{n}_j $$

- Dzyaloshinskii–Moriya interaction (antisymmetric exchange)
$$ \mathcal{H}_{DMI}[\vec{n}] = - \sum\limits_{\braket{ij}}\, \vec{D}_{ij} \cdot (\vec{n}_i\times\vec{n}_j) $$

where it is important to note that `<ij>` denotes the unique pairs of interacting spins `i` and `j`.

---

### Dipole-Diplole Interaction
$$
\mathcal{H}_{DDI}[\vec{n}]
    = \frac{1}{2}\frac{\mu_0}{4\pi} \sum_{\substack{i,j \\ i \neq j}} \mu_i \mu_j \frac{(\vec{n}_i \cdot \hat{r}_{ij}) (\vec{n}_j\cdot\hat{r}_{ij}) - \vec{n}_i \vec{n}_j}{{r_{ij}}^3}
$$

---

### Quadruplet Interaction
$$ \mathcal{H}_{Quad}[\vec{n}] = - \sum_{i,j,k,l} K_{ijkl} (\vec{n}_i \cdot \vec{n}_j)(\vec{n}_k \cdot \vec{n}_l) $$

---

<!-- ![](https://math.now.sh?from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright)) -->


### Gaussian (test-) Hamiltonian

The Hamiltonian is defined as
$$ \mathcal{H}[\vec{n}] = \sum\limits_i \mathcal{H}_i[\vec{n}]  = \sum\limits_i a_i \exp\left( -\frac{(1 - \vec{n}\cdot\vec{c}_i)^2}{2\sigma_i^2} \right) $$

<!-- ![](https://math.now.sh?from=%5Cmathcal%7BH%7D%20%3D%20%5Csum%5Climits_i%20%5Cmathcal%7BH%7D_i%20%20%3D%20%5Csum%5Climits_i%20a_i%20%5Cexp%5Cleft%28%20-%5Cfrac%7B(1%20-%20%5Cvec%7Bn%7D%5Ccdot%5Cvec%7Bc%7D_i%29%5E2%7D%7B2%5Csigma_i%5E2%7D%20%5Cright)) -->
