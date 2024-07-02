Heisenberg Hamiltonian
====================================================

### Hamiltonian
The Spin-Hamiltonian is defined as

![Hamiltonian](https://math.vercel.app/?from=%5Cmathcal%7BH%7D%5B%5Cvec%7Bn%7D%5D%20%3D%0A%20%20%20%20%5Cmathcal%7BH%7D_%7BZ%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%2B%20%5Cmathcal%7BH%7D_%7BA%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%2B%20%5Cmathcal%7BH%7D_%7BXC%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%2B%20%5Cmathcal%7BH%7D_%7BDMI%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%2B%20%5Cmathcal%7BH%7D_%7BDDI%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%2B%20%5Cmathcal%7BH%7D_%7BQuad%7D%5B%5Cvec%7Bn%7D%5D%0A.svg)

<!-- $$
\mathcal{H}[\vec{n}] =
    \mathcal{H}_{Z}[\vec{n}]
    + \mathcal{H}_{A}[\vec{n}]
    + \mathcal{H}_{XC}[\vec{n}]
    + \mathcal{H}_{DMI}[\vec{n}]
    + \mathcal{H}_{DDI}[\vec{n}]
    + \mathcal{H}_{Quad}[\vec{n}]
$$ -->

---

### Zeeman

![Zeeman](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_Z%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum_i%20%5Cmu_i%20%5Cvec%7BB%7D%5Ccdot%5Cvec%7Bn%7D_i.svg)

<!-- $$ \mathcal{H}_Z[\vec{n}] = - \sum_i \mu_i \vec{B}\cdot\vec{n}_i $$ -->

---

### Anisotropy
The anisotropy term is implemented in terms of three components:
- uniaxial anisotropy

![Uniaxial Anisotropy](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_%7BA_1%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum_j%20K_j%20%28%5Chat%7BK%7D_j%5Ccdot%5Cvec%7Bn%7D_j%29%5E2%20%3D%20-%20%5Csum_j%20K_j%20%5B%5Ccos%28%5Ctheta_j%29%5D%5E2.svg)

<!-- $$ \mathcal{H}_{A_1}[\vec{n}] = - \sum_j K_j (\hat{K}_j\cdot\vec{n}_j)^2 = - \sum_j K_j [\cos(\theta_j)]^2 $$ -->

- cubic anisotropy

![Cubic Anisotropy](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_%7BA_c%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_j%20K_j%20%28%5B%5Cvec%7Bn%7D_j%5D_x%5E4%20%2B%20%5B%5Cvec%7Bn%7D_j%5D_y%5E4%20%2B%20%5B%5Cvec%7Bn%7D_j%5D_z%5E4%29.svg)

<!-- $$ \mathcal{H}_{A_c}[\vec{n}] = - \frac{1}{2} \sum_j K_j ([\vec{n}_j]_x^4 + [\vec{n}_j]_y^4 + [\vec{n}_j]_z^4) $$ -->

- biaxial anisotropy

![Biaxial Anisotropy](https://math.vercel.app/?from=%5Cbegin%7Balignedat%7D%7B1%7D%0A%5Cmathcal%7BH%7D_%7BA_2%7D%20%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%281%20-%20%28%5Chat%7BK%7D_j%5E%7B%281%29%7D%5Ccdot%5Cvec%7Bn%7D_j%29%5E2%29%5E%7Bn_1%7D%20%5Ccdot%20%28%5Chat%7BK%7D_j%5E%7B%282%29%7D%5Ccdot%5Cvec%7Bn%7D_j%29%5E%7Bn_2%7D%20%5Ccdot%20%28%28%5Chat%7BK%7D_j%5E%7B%281%29%7D%20%5Ctimes%20%5Chat%7BK%7D_j%5E%7B%282%29%7D%20%29%20%5Ccdot%5Cvec%7Bn%7D_j%29%5E%7Bn_3%7D%20%5C%5C%0A%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%5Ccdot%20%5B%5Csin%28%5Ctheta_j%29%5D%5E%7B2n_1%7D%20%5Ccdot%20%5B%5Ccos%28%5Cvarphi_j%29%5Csin%28%5Ctheta_j%29%5D%5E%7Bn_2%7D%20%5Ccdot%20%5B%5Csin%28%5Cvarphi_j%29%5Csin%28%5Ctheta_j%29%5D%5E%7Bn_3%7D%20%5C%5C%0A%3D%26%20%5Csum_%7Bj%7D%20%5Csum_%7Bn_1%2Cn_2%2Cn_3%7D%20K_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%5Ccdot%20%5B%5Csin%28%5Ctheta_j%29%5D%5E%7B2n_1%20%2B%20n_2%20%2B%20n_3%7D%20%5Ccdot%20%5B%5Ccos%28%5Cvarphi_j%29%5D%5E%7Bn_2%7D%20%5Ccdot%20%5B%5Csin%28%5Cvarphi_j%29%5D%5E%7Bn_3%7D%20%5C%5C%0A%5Cend%7Balignedat%7D.svg)

<!-- $$
\begin{alignedat}{1}
\mathcal{H}_{A_2} =& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} (1 - (\hat{K}_j^{(1)}\cdot\vec{n}_j)^2)^{n_1} \cdot (\hat{K}_j^{(2)}\cdot\vec{n}_j)^{n_2} \cdot ((\hat{K}_j^{(1)} \times \hat{K}_j^{(2)} ) \cdot\vec{n}_j)^{n_3} \\
=& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} \cdot [\sin(\theta_j)]^{2n_1} \cdot [\cos(\varphi_j)\sin(\theta_j)]^{n_2} \cdot [\sin(\varphi_j)\sin(\theta_j)]^{n_3} \\
=& \sum_{j} \sum_{n_1,n_2,n_3} K_j^{(n_1, n_2, n_3)} \cdot [\sin(\theta_j)]^{2n_1 + n_2 + n_3} \cdot [\cos(\varphi_j)]^{n_2} \cdot [\sin(\varphi_j)]^{n_3} \\
\end{alignedat}
$$ -->

Where for any site $j$ the vectors $\hat{K}_j^{(1)}$, $\hat{K}_j^{(2)}$ and $\hat{K}_j^{(3)} = \hat{K}_j^{(1)} \times \hat{K}_j^{(2)}$ are pairwise orthonormal.


The uniaxial anisotropy is equivalent to the biaxial anisotropy up to an offset to the total energy when setting

![equivalence](https://math.vercel.app/?from=%5Cbegin%7Balignedat%7D%7B1%7D%0A%5Chat%7BK%7D_j%5E%7B%281%29%7D%20%3D%26%20%5Chat%7BK%7D_j%20%5C%5C%0AK_j%5E%7B%28n_1%2C%20n_2%2C%20n_3%29%7D%20%3D%26%20-K_j%20%5Cdelta_%7Bn_1%2C%201%7D%5Cdelta_%7Bn_2%2C%200%7D%5Cdelta_%7Bn_3%2C%200%7D.%0A%5Cend%7Balignedat%7D%0A.svg)

<!-- $$
\begin{alignedat}{1}
\hat{K}_j^{(1)} =& \hat{K}_j \\
K_j^{(n_1, n_2, n_3)} =& -K_j \delta_{n_1, 1}\delta_{n_2, 0}\delta_{n_3, 0}.
\end{alignedat}
$$ -->

---

### Exchange
- symmetric exchange interaction

![Exchange](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_%7BXC%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20J_%7Bij%7D%20%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j.svg)

<!-- $$ \mathcal{H}_{XC}[\vec{n}] = - \sum\limits_{\braket{ij}}\, J_{ij} \vec{n}_i\cdot\vec{n}_j $$ -->

- Dzyaloshinskii–Moriya interaction (antisymmetric exchange)

![DMI](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_%7BDMI%7D%5B%5Cvec%7Bn%7D%5D%20%3D%20-%20%5Csum%5Climits_%7B%5Cbraket%7Bij%7D%7D%5C%2C%20%5Cvec%7BD%7D_%7Bij%7D%20%5Ccdot%20%28%5Cvec%7Bn%7D_i%5Ctimes%5Cvec%7Bn%7D_j%29%0A.svg)

<!-- $$ \mathcal{H}_{DMI}[\vec{n}] = - \sum\limits_{\braket{ij}}\, \vec{D}_{ij} \cdot (\vec{n}_i\times\vec{n}_j) $$ -->

where it is important to note that `<ij>` denotes the unique pairs of interacting spins `i` and `j`.

---

### Dipole-Dipole Interaction

![Dipole-Dipole Interaction](https://math.vercel.app/?from=%5Cmathcal%7BH%7D_%7BDDI%7D%5B%5Cvec%7Bn%7D%5D%0A%20%20%20%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cmu_0%7D%7B4%5Cpi%7D%20%5Csum_%7B%5Csubstack%7Bi%2Cj%20%5C%5C%20i%20%5Cneq%20j%7D%7D%20%5Cmu_i%20%5Cmu_j%20%5Cfrac%7B%28%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Chat%7Br%7D_%7Bij%7D%29%20%28%5Cvec%7Bn%7D_j%5Ccdot%5Chat%7Br%7D_%7Bij%7D%29%20-%20%5Cvec%7Bn%7D_i%20%5Cvec%7Bn%7D_j%7D%7B%7Br_%7Bij%7D%7D%5E3%7D%0A.svg)

<!-- $$
\mathcal{H}_{DDI}[\vec{n}]
    = \frac{1}{2}\frac{\mu_0}{4\pi} \sum_{\substack{i,j \\ i \neq j}} \mu_i \mu_j \frac{(\vec{n}_i \cdot \hat{r}_{ij}) (\vec{n}_j\cdot\hat{r}_{ij}) - \vec{n}_i \vec{n}_j}{{r_{ij}}^3}
$$ -->

---

### Quadruplet Interaction

![Quadruplet Interaction](https://math.now.sh?from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright))

<!-- $$ \mathcal{H}_{Quad}[\vec{n}] = - \sum_{i,j,k,l} K_{ijkl} (\vec{n}_i \cdot \vec{n}_j)(\vec{n}_k \cdot \vec{n}_l) $$ -->

---



### Gaussian (test-) Hamiltonian

The Hamiltonian is defined as

![Gaussian Hamiltonian](https://math.now.sh?from=%5Cmathcal%7BH%7D%20%3D%20%5Csum%5Climits_i%20%5Cmathcal%7BH%7D_i%20%20%3D%20%5Csum%5Climits_i%20a_i%20%5Cexp%5Cleft%28%20-%5Cfrac%7B(1%20-%20%5Cvec%7Bn%7D%5Ccdot%5Cvec%7Bc%7D_i%29%5E2%7D%7B2%5Csigma_i%5E2%7D%20%5Cright))

<!-- $$ \mathcal{H}[\vec{n}] = \sum\limits_i \mathcal{H}_i[\vec{n}]  = \sum\limits_i a_i \exp\left( -\frac{(1 - \vec{n}\cdot\vec{c}_i)^2}{2\sigma_i^2} \right) $$ -->
