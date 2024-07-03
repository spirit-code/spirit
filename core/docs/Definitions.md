Definitions
====================================================


All definitions are described in detail in [G. P. Müller *et al.*, Phys. Rev. B **99**, 224414 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.224414).
Here we make brief summaries to give you an overview.


Heisenberg Hamiltonian
----------------------------------------------------

The Hamiltonian is defined as

<!-- \begin{alignedat}{1}
\mathcal{H} =
    - \sum_i \mu_i \vec{B}\cdot\vec{n}_i
    - \sum_i \sum_j K_j (\hat{K}_j\cdot\vec{n}_i)^2
    - \sum\limits_{\braket{ij}}\, J_{ij} \vec{n}_i\cdot\vec{n}_j
    - \sum\limits_{\braket{ij}}\, \vec{D}_{ij} \cdot (\vec{n}_i\times\vec{n}_j)
    + \frac{1}{2}\frac{\mu_0}{4\pi} \sum_{\substack{i,j \\ i \neq j}} \mu_i \mu_j \frac{(\vec{n}_i \cdot \hat{r}_{ij}) (\vec{n}_j\cdot\hat{r}_{ij}) - \vec{n}_i \vec{n}_j}{{r_{ij}}^3}
\end{alignedat} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D%20%3D%0A%20%20%20%20-%20%5Csum_i%20%5Cmu_i%20%5Cvec%7BB%7D%5Ccdot%5Cvec%7Bn%7D_i%0A%20%20%20%20-%20%5Csum_i%20%5Csum_j%20K_j%20%28%5Chat%7BK%7D_j%5Ccdot%5Cvec%7Bn%7D_i%29%5E2%0A%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7B%5C%3B%20ij%7D%7D%5C%2C%20J_%7Bij%7D%20%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%0A%20%20%20%20-%20%5Csum%5Climits_%7B%5Cbraket%7B%5C%3Bij%7D%7D%5C%2C%20%5Cvec%7BD%7D_%7Bij%7D%20%5Ccdot%20(%5Cvec%7Bn%7D_i%5Ctimes%5Cvec%7Bn%7D_j)%0A%20%20%20%20%2B%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cmu_0%7D%7B4%5Cpi%7D%20%5Csum_%7B%5Csubstack%7Bi%2Cj%20%5C%5C%20i%20%5Cneq%20j%7D%7D%20%5Cmu_i%20%5Cmu_j%20%5Cfrac%7B(%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Chat%7Br%7D_%7Bij%7D)%20(%5Cvec%7Bn%7D_j%5Ccdot%5Chat%7Br%7D_%7Bij%7D)%20-%20%5Cvec%7Bn%7D_i%20%5Cvec%7Bn%7D_j%7D%7B%7Br_%7Bij%7D%7D%5E3%7D)

where it is important to note that `<ij>` denotes the unique pairs of interacting spins `i` and `j`.

The quadruplet interaction is defined as

![](https://math.vercel.app/?bgcolor=auto&from=E_%5Cmathrm%7BQuad%7D%20%3D%20-%20%5Csum%5Climits_%7Bijkl%7D%5C%2C%20K_%7Bijkl%7D%20%5Cleft%28%5Cvec%7Bn%7D_i%5Ccdot%5Cvec%7Bn%7D_j%5Cright%29%5Cleft(%5Cvec%7Bn%7D_k%5Ccdot%5Cvec%7Bn%7D_l%5Cright))


LLG dynamics
--------------------------------------------------


Spirit denotes the LLG equation as

<!-- \begin{alignedat}{2}
\dfrac{\partial \vec{n}_i}{\partial t}
    =& - \dfrac{\gamma}{(1+\alpha^2)\mu_i} \vec{n}_i \times \vec{B}^\mathrm{eff}_i
    - \dfrac{\gamma \alpha}{(1+\alpha^2)\mu_i} \vec{n}_i \times (\vec{n}_i \times \vec{B}^\mathrm{eff}_i) \\
    &- \dfrac{\alpha-\beta}{(1+\alpha^2)} u \vec{n}_i \times (\hat{j}_e \cdot \nabla_{\vec{r}} )\vec{n}_i
    + \dfrac{1+\beta \alpha}{(1+\alpha^2)} u \vec{n}_i \times (\vec{n}_i \times (\hat{j}_e \cdot \nabla_{\vec{r}} )\vec{n}_i)
\end{alignedat} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cbegin%7Balignedat%7D%7B2%7D%0A%20%20%20%20%5Cdfrac%7B%5Cpartial%20%5Cvec%7Bn%7D_i%7D%7B%5Cpartial%20t%7D%0A%20%20%20%20%20%20%20%20%3D%26%20-%20%5Cdfrac%7B%5Cgamma%7D%7B%281%2B%5Calpha%5E2%29%5Cmu_i%7D%20%5Cvec%7Bn%7D_i%20%5Ctimes%20%5Cvec%7BB%7D%5E%5Cmathrm%7Beff%7D_i%20%0A%20%20%20%20%20%20%20%20-%20%5Cdfrac%7B%5Cgamma%20%5Calpha%7D%7B(1%2B%5Calpha%5E2)%5Cmu_i%7D%20%5Cvec%7Bn%7D_i%20%5Ctimes%20(%5Cvec%7Bn%7D_i%20%5Ctimes%20%5Cvec%7BB%7D%5E%5Cmathrm%7Beff%7D_i)%20%5C%5C%0A%20%20%20%20%20%20%20%20%26-%20%5Cdfrac%7B%5Calpha-%5Cbeta%7D%7B(1%2B%5Calpha%5E2)%7D%20u%20%5Cvec%7Bn%7D_i%20%5Ctimes%20(%5Chat%7Bj%7D_e%20%5Ccdot%20%5Cnabla_%7B%5Cvec%7Br%7D%7D%20)%5Cvec%7Bn%7D_i%0A%20%20%20%20%20%20%20%20%2B%20%5Cdfrac%7B1%2B%5Cbeta%20%5Calpha%7D%7B(1%2B%5Calpha%5E2)%7D%20u%20%5Cvec%7Bn%7D_i%20%5Ctimes%20(%5Chat%7Bn%7D_i%20%5Ctimes%20(%5Chat%7Bj%7D_e%20%5Ccdot%20%5Cnabla_%7B%5Cvec%7Br%7D%7D%20)%5Cvec%7Bn%7D_i)%0A%5Cend%7Balignedat%7D)


γ is the electron gyromagnetic ratio, α is the damping parameter, β is a non-adiabaticity parameter, with

<!-- \nabla_{\vec{r}} = \partial / \partial \vec{r} -->
![](https://math.vercel.app/?bgcolor=auto&from=u%3Dj_e%20P%20g%20%5Cmu_%5Cmathrm%7BB%7D%2F%282eM_%5Cmathrm%7BS%7D%29)
and
![](https://math.vercel.app/?bgcolor=auto&from=%5Cnabla_%7B%5Cvec%7Br%7D%7D%20%3D%20%5Cpartial%20%2F%20%5Cpartial%20%5Cvec%7Br%7D)


If temperature is used, a thermal component is added to the effective magnetic field:

<!-- \vec{B}^\mathrm{th}_i(t) = \sqrt{2D_i} \vec{\eta}_i(t) = \sqrt{2\alpha k_\mathrm{B}T \frac{\mu_i}{\gamma}} \vec{\eta}_i(t) -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cvec%7BB%7D%5E%5Cmathrm%7Bth%7D_i%28t%29%20%3D%20%5Csqrt%7B2D_i%7D%20%5Cvec%7B%5Ceta%7D_i(t)%20%3D%20%5Csqrt%7B2%5Calpha%20k_%5Cmathrm%7BB%7DT%20%5Cfrac%7B%5Cmu_i%7D%7B%5Cgamma%7D%7D%20%5Cvec%7B%5Ceta%7D_i(t))


Geodesic nudged elastic band method
--------------------------------------------------


The total force is

<!-- F^\mathrm{tot}_\nu = F^\mathrm{S}_\nu + F^\mathrm{E}_\nu -->
![](https://math.vercel.app/?bgcolor=auto&from=F%5E%5Cmathrm%7Btot%7D_%5Cnu%20%3D%20F%5E%5Cmathrm%7BS%7D_%5Cnu%20%2B%20F%5E%5Cmathrm%7BE%7D_%5Cnu)

with the spring force

<!-- F^\mathrm{S}_\nu = (l_{\nu-1,\nu}-l_{\nu,\nu+1})\ \tau_\nu -->
![](https://math.vercel.app/?bgcolor=auto&from=F%5E%5Cmathrm%7BS%7D_%5Cnu%20%3D%20%28l_%7B%5Cnu-1%2C%5Cnu%7D-l_%7B%5Cnu%2C%5Cnu%2B1%7D%29%5C%20%5Ctau_%5Cnu)

and the energy gradient force

<!-- F^\mathrm{E}_\nu = -\nabla E_\nu + (\nabla E_\nu \cdot \tau_\nu)\tau_\nu -->
![](https://math.vercel.app/?bgcolor=auto&from=F%5E%5Cmathrm%7BE%7D_%5Cnu%20%3D%20-%5Cnabla%20E_%5Cnu%20%2B%20%28%5Cnabla%20E_%5Cnu%20%5Ccdot%20%5Ctau_%5Cnu%29%5Ctau_%5Cnu)


The corresponding 3-component subvectors need to be orthogonalized with respect to the spins:

<!-- \vec{\tau}_{\nu,i} \to \vec{\tau}_{\nu,i} - (\vec{\tau}_{\nu,i}\cdot \vec{n}_{\nu,i})\vec{n}_{\nu,i} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cvec%7B%5Ctau%7D_%7B%5Cnu%2Ci%7D%20%5Cto%20%5Cvec%7B%5Ctau%7D_%7B%5Cnu%2Ci%7D%20-%20%28%5Cvec%7B%5Ctau%7D_%7B%5Cnu%2Ci%7D%5Ccdot%20%5Cvec%7Bn%7D_%7B%5Cnu%2Ci%7D%29%5Cvec%7Bn%7D_%7B%5Cnu%2Ci%7D)

The spring forces need to be projected as well

<!-- \vec{F}^\mathrm{E}_{\nu,i} \to \vec{F}^\mathrm{E}_{\nu,i}  - (\vec{F}^\mathrm{E}_{\nu,i} \cdot \vec{n}_{\nu,i}) \vec{n}_{\nu,i} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cvec%7BF%7D%5E%5Cmathrm%7BE%7D_%7B%5Cnu%2Ci%7D%20%5Cto%20%5Cvec%7BF%7D%5E%5Cmathrm%7BE%7D_%7B%5Cnu%2Ci%7D%20%20-%20%28%5Cvec%7BF%7D%5E%5Cmathrm%7BE%7D_%7B%5Cnu%2Ci%7D%20%5Ccdot%20%5Cvec%7Bn%7D_%7B%5Cnu%2Ci%7D%29%20%5Cvec%7Bn%7D_%7B%5Cnu%2Ci%7D)


*Note the features of "climbing/falling images" and "path shortening", which are described in detail in the paper.*


Minimum mode following method
--------------------------------------------------


The mode following force is given by an inversion of the energy gradient force along the mode λ:

<!-- F^\mathrm{eff} = F - 2 (F\cdot{\hat\lambda}) {\hat\lambda} -->
![](https://math.vercel.app/?bgcolor=auto&from=F%5E%5Cmathrm%7Beff%7D%20%3D%20F%20-%202%20%28F%5Ccdot%7B%5Chat%5Clambda%7D%29%20%7B%5Chat%5Clambda%7D)

To calculate the energy eigenmodes, we calculate the Hessian matrix

<!-- H_{ij} = T_i^T \bar{H}_{ij} T_j - T_i^T I (\vec{n}_j\cdot\vec{\nabla}_j\bar{\mathcal{H}}) T_j -->
![](https://math.vercel.app/?bgcolor=auto&from=H_%7Bij%7D%20%3D%20T_i%5ET%20%5Cbar%7BH%7D_%7Bij%7D%20T_j%20-%20T_i%5ET%20I%20%28%5Cvec%7Bn%7D_j%5Ccdot%5Cvec%7B%5Cnabla%7D_j%5Cbar%7B%5Cmathcal%7BH%7D%7D%29%20T_j)

Details on this method, the equations and their derivations can be found in [G. P. Müller *et al.*, Phys. Rev. Lett. **121**, 197202](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.197202) and [[1]](#Thesis).


Harmonic transition state theory
--------------------------------------------------


The transition rate reads

<!-- \Gamma^\mathrm{HTST} = \frac{v}{2\pi} \Omega_0 e^{-\Delta E/k_\mathrm{B}T} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5CGamma%5E%5Cmathrm%7BHTST%7D%20%3D%20%5Cfrac%7Bv%7D%7B2%5Cpi%7D%20%5COmega_0%20e%5E%7B-%5CDelta%20E%2Fk_%5Cmathrm%7BB%7DT%7D)

with

<!-- \Omega_0
    = \sqrt{\frac{\det^\prime H^\mathrm{M}}{\det^\prime H^\mathrm{S}}}
    = \sqrt{\frac{\sideset{}{'}\prod_i \lambda_i^\mathrm{M}}{\sideset{}{'}\prod_i \lambda_i^\mathrm{S}}} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5COmega_0%0A%20%20%20%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Cdet%5E%5Cprime%20H%5E%5Cmathrm%7BM%7D%7D%7B%5Cdet%5E%5Cprime%20H%5E%5Cmathrm%7BS%7D%7D%7D%0A%20%20%20%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csideset%7B%7D%7B'%7D%5Cprod_i%20%5Clambda_i%5E%5Cmathrm%7BM%7D%7D%7B%5Csideset%7B%7D%7B'%7D%5Cprod_i%20%5Clambda_i%5E%5Cmathrm%7BS%7D%7D%7D)

<!-- v
    = \sqrt{ 2\pi k_\mathrm{B}T }^{N_0^\mathrm{M} - N_0^\mathrm{S}}
    \frac{V^\mathrm{S}}{V^\mathrm{M}}
    \sqrt{\sideset{}{'}\sum_i \frac{a_i^2}{\lambda_i^\mathrm{S}}} -->
![](https://math.vercel.app/?bgcolor=auto&from=v%0A%20%20%20%20%3D%20%5Csqrt%7B%202%5Cpi%20k_%5Cmathrm%7BB%7DT%20%7D%5E%7BN_0%5E%5Cmathrm%7BM%7D%20-%20N_0%5E%5Cmathrm%7BS%7D%7D%0A%20%20%20%20%5Cfrac%7BV%5E%5Cmathrm%7BS%7D%7D%7BV%5E%5Cmathrm%7BM%7D%7D%0A%20%20%20%20%5Csqrt%7B%5Csideset%7B%7D%7B'%7D%5Csum_i%20%5Cfrac%7Ba_i%5E2%7D%7B%5Clambda_i%5E%5Cmathrm%7BS%7D%7D%7D)

Details on these equations and their derivations can be found in [[1]](#Thesis).


Topological charge
--------------------------------------------------


The topological charge is defined as

<!-- Q = \frac{1}{4\pi} \int_{\mathbb{R}^2} \vec{n} \cdot (\partial_x \vec{n} \times \partial_y \vec{n})\, \mathrm{d}\vec{r} -->
![](https://math.vercel.app/?bgcolor=auto&from=Q%20%3D%20%5Cfrac%7B1%7D%7B4%5Cpi%7D%20%5Cint_%7B%5Cmathbb%7BR%7D%5E2%7D%20%5Cvec%7Bn%7D%20%5Ccdot%20%28%5Cpartial_x%20%5Cvec%7Bn%7D%20%5Ctimes%20%5Cpartial_y%20%5Cvec%7Bn%7D%29%5C%2C%20%5Cmathrm%7Bd%7D%5Cvec%7Br%7D)

On a discrete lattice, this corresponds to

<!-- Q= \frac{1}{4\pi}\sum_l A_l -->
![](https://math.vercel.app/?bgcolor=auto&from=Q%3D%20%5Cfrac%7B1%7D%7B4%5Cpi%7D%5Csum_l%20A_l)

with

<!-- \cos\left(\frac{A_l}{2}\right)=\frac{1  + \vec{n}_i \cdot \vec{n}_j + \vec{n}_i \cdot \vec{n}_k + \vec{n}_j \cdot \vec{n}_k}
    {\sqrt{2\left(1+\vec{n}_i\vec{n}_j\right)\left(1+\vec{n}_j\vec{n}_k
    \right)\left(1+\vec{n}_k\vec{n}_i\right)}} -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Ccos%5Cleft%28%5Cfrac%7BA_l%7D%7B2%7D%5Cright%29%3D%5Cfrac%7B1%20%20%2B%20%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Cvec%7Bn%7D_j%20%2B%20%5Cvec%7Bn%7D_i%20%5Ccdot%20%5Cvec%7Bn%7D_k%20%2B%20%5Cvec%7Bn%7D_j%20%5Ccdot%20%5Cvec%7Bn%7D_k%7D%0A%20%20%20%20%7B%5Csqrt%7B2%5Cleft(1%2B%5Cvec%7Bn%7D_i%5Cvec%7Bn%7D_j%5Cright)%5Cleft(1%2B%5Cvec%7Bn%7D_j%5Cvec%7Bn%7D_k%0A%20%20%20%20%5Cright)%5Cleft(1%2B%5Cvec%7Bn%7D_k%5Cvec%7Bn%7D_i%5Cright)%7D%7D)


Gaussian (test-) Hamiltonian
--------------------------------------------------


The Hamiltonian is defined as

<!-- \mathcal{H} = \sum\limits_i \mathcal{H}_i  = \sum\limits_i a_i \exp\left( -\frac{(1 - \vec{n}\cdot\vec{c}_i)^2}{2\sigma_i^2} \right) -->
![](https://math.vercel.app/?bgcolor=auto&from=%5Cmathcal%7BH%7D%20%3D%20%5Csum%5Climits_i%20%5Cmathcal%7BH%7D_i%20%20%3D%20%5Csum%5Climits_i%20a_i%20%5Cexp%5Cleft%28%20-%5Cfrac%7B(1%20-%20%5Cvec%7Bn%7D%5Ccdot%5Cvec%7Bc%7D_i%29%5E2%7D%7B2%5Csigma_i%5E2%7D%20%5Cright))


--------------------------------------------------


[1]:<a name="Thesis"></a> **G. P. Müller, Advanced methods for atomic scale spin simulations and application to localized magnetic states. PhD Thesis (2019)** (availavle from [Univ. of Iceland](https://opinvisindi.is/handle/20.500.11815/1256), [RWTH Aachen](https://publications.rwth-aachen.de/record/767445) and [FZ Jülich](https://juser.fz-juelich.de/record/866248))