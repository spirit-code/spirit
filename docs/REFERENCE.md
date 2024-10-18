# Reference

The **Spirit** framework is a scientific project.
If you present and/or publish scientific results or visualizations that used **Spirit**, please always cite [the Spirit paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.224414), the (version-independent) [Zenodo entry](https://doi.org/10.5281/zenodo.7746552) for **Spirit**, and mention the version.

### Framework

When referring to **Spirit** as a scientific project, please Cite:

    @article{spiritPaper,
      title = {Spirit: Multifunctional framework for atomistic spin simulations},
      author = {M\"uller, Gideon P. and Hoffmann, Markus and Di\ss{}elkamp, Constantin and Sch\"urhoff, Daniel and Mavros, Stefanos and Sallermann, Moritz and Kiselev, Nikolai S. and J\'onsson, Hannes and Bl\"ugel, Stefan},
      journal = {Phys. Rev. B},
      volume = {99},
      issue = {22},
      pages = {224414},
      numpages = {16},
      year = {2019},
      month = {Jun},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevB.99.224414},
      url = {https://link.aps.org/doi/10.1103/PhysRevB.99.224414}
    }

- G. P. Müller, M. Hoffmann, C. Disselkamp, D. Schürhoff, S. Mavros, M. Sallermann, N. S. Kiselev, H. Jónsson, S. Blügel. "Spirit: Multifunctional framework for atomistic spin simulations." Phys. Rev. B 99, 224414 (2019)

If you used **Spirit** to produce scientific results, please cite

    @misc{spiritCode,
      author       = {Müller, Gideon P. and Sallermann, Moritz and Mavros, Stefanos and Rhiem, Florian and Schürhoff, Daniel and Hoffmann, Markus and Meyer, Ingo and Disselkamp, Constantin and Redies, Matthias and Buhl, Patrick and Suckert, Jens Rene and Ivanov, Aleksei V. and Kiselev, Nikolai S. and Jónsson, Hannes and Blügel, Stefan},
      title        = {{SPIRIT}},
      month        = mar,
      year         = 2023,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.7746552},
      url          = {https://doi.org/10.5281/zenodo.7746552},
      howpublished  = {Zenodo}
    }

- G. P. Müller, M. Sallermann, S. Mavros, F. Rhiem, D. Schürhoff, M. Hoffmann, I. Meyer, C. Disselkamp, M. Redies, P. Buhl, J. R. Suckert, A. V. Ivanov, N. S. Kiselev, H. Jónsson, S. Blügel (2023). spirit-code/spirit: New Desktop GUI, update to C++14 (v2.2.0). Zenodo. https://doi.org/10.5281/zenodo.7746552

You may need to update the DOI and Zenodo URL to the specific version you were using.

### Source Code

When referring to specific sections of code you may also reference our GitHub page.
You may use e.g. the following TeX code:

    @misc{spiritWeb,
      author = {},
      title = {{Spirit spin simulation framework}},
      howpublished = {\url{https://spirit-code.github.io}}
    }

- Spirit spin simulation framework (see https://spirit-code.github.io)

### Specific Methods

The following need only be cited if used.

**Depondt Solver**

This Heun-like method for solving the LLG equation including the
stochastic term has been published by Depondt et al.:
http://iopscience.iop.org/0953-8984/21/33/336005
You may use e.g. the following TeX code:

    \bibitem{Depondt}
    Ph. Depondt et al. \textit{J. Phys. Condens. Matter} \textbf{21}, 336005 (2009).

**SIB Solver**

This stable method for solving the LLG equation efficiently and
including the stochastic term has been published by Mentink et al.:
http://iopscience.iop.org/0953-8984/22/17/176001
You may use e.g. the following TeX code:

    \bibitem{SIB}
    J. H. Mentink et al. \textit{J. Phys. Condens. Matter} \textbf{22}, 176001 (2010).

**VP Solver**

This intuitive direct minimization routine has been published as
supplementary material by Bessarab et al.:
http://www.sciencedirect.com/science/article/pii/S0010465515002696
You may use e.g. the following TeX code:

    \bibitem{VP}
    P. F. Bessarab et al. \textit{Comp. Phys. Comm.} \textbf{196}, 335 (2015).

**GNEB Method**

This specialized nudged elastic band method for calculating transition
paths of spin systems has been published by Bessarab et al.:
http://www.sciencedirect.com/science/article/pii/S0010465515002696
You may use e.g. the following TeX code:

    \bibitem{GNEB}
    P. F. Bessarab et al. \textit{Comp. Phys. Comm.} \textbf{196}, 335 (2015).

**HTST**

The harmonic transition state theory for calculating transition
rates of spin systems has been published by Bessarab et al.:
https://link.aps.org/doi/10.1103/PhysRevB.85.184409
You may use e.g. the following TeX code:

    \bibitem{GNEB}
    P. F. Bessarab et al. \textit{Comp. Phys. Comm.} \textbf{196}, 335 (2015).

**MMF Method**

The mode following method, intended for saddle point searches,
has been published by Müller et al.:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.197202
You may use e.g. the following TeX code:

    \bibitem{MMF}
    G. P. Müller et al. Phys. Rev. Lett. 121, 197202 (2018).


---

[Home](README.md)
