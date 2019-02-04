# Reference

The **Spirit** framework is a scientific project.
If you present and/or publish scientific results or
visualisations that used Spirit, you should add a reference.


### The Framework

If you used Spirit to produce scientific results or when referring to it as a
scientific project, please cite the paper.

    \bibitem{mueller_spirit_2019}{
        G. P. Müller, M. Hoffmann, C. Disselkamp, D. Schürhoff, S. Mavros, M. Sallermann, N. S. Kiselev, H. Jónsson, S. Blügel.
        "Spirit: Multifunctional Framework for Atomistic Spin Simulations."
        arXiv:1901.11350
    }

When referring to code of this framework please add a reference to our GitHub page.
You may use e.g. the following TeX code:

    \bibitem{spirit}
    {Spirit spin simulation framework} (see spirit-code.github.io)


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

**MMF Method**

The mode following method, intended for saddle point searches,
has been published by Müller et al.:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.197202
You may use e.g. the following TeX code:

    \bibitem{MMF}
    G. P. Müller et al. Phys. Rev. Lett. 121, 197202 (2018).


---

[Home](Readme.md)