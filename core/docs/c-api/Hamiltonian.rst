Hamiltonian
=========================

.. code-block:: C

  #include "Spirit/Hamiltonian.h"

This currently only provides an interface to the Heisenberg Hamiltonian.


DMI chirality
--------------------------------------------------------------------

.. doxygendefine:: SPIRIT_CHIRALITY_BLOCH 1
.. doxygendefine:: SPIRIT_CHIRALITY_NEEL 2
.. doxygendefine:: SPIRIT_CHIRALITY_BLOCH_INVERSE -1
.. doxygendefine:: SPIRIT_CHIRALITY_NEEL_INVERSE -2


Dipole-Dipole method
--------------------------------------------------------------------

.. doxygendefine:: SPIRIT_DDI_METHOD_NONE 0
.. doxygendefine:: SPIRIT_DDI_METHOD_FFT 1
.. doxygendefine:: SPIRIT_DDI_METHOD_FMM 2
.. doxygendefine:: SPIRIT_DDI_METHOD_CUTOFF 3


Hamiltonian classification id (legacy support)
--------------------------------------------------------------------

.. doxygendefine:: SPIRIT_HAMILTONIAN_CLASS_GENERIC 0
.. doxygendefine:: SPIRIT_HAMILTONIAN_CLASS_GAUSSIAN 1
.. doxygendefine:: SPIRIT_HAMILTONIAN_CLASS_HEISENBERG 2


Setters
--------------------------------------------------------------------

.. doxygenfunction:: Hamiltonian_Set_Boundary_Conditions
.. doxygenfunction:: Hamiltonian_Set_Field
.. doxygenfunction:: Hamiltonian_Set_Anisotropy
.. doxygenfunction:: Hamiltonian_Set_Biaxial_Anisotropy
.. doxygenfunction:: Hamiltonian_Set_Exchange
.. doxygenfunction:: Hamiltonian_Set_DMI
.. doxygenfunction:: Hamiltonian_Set_DDI


Getters
--------------------------------------------------------------------


.. doxygenfunction:: Hamiltonian_Get_Boundary_Conditions
.. doxygenfunction:: Hamiltonian_Get_Field
.. doxygenfunction:: Hamiltonian_Get_Anisotropy
.. doxygenfunction:: Hamiltonian_Get_Biaxial_Anisotropy_N_Atoms
.. doxygenfunction:: Hamiltonian_Get_Biaxial_Anisotropy_N_Terms
.. doxygenfunction:: Hamiltonian_Get_Biaxial_Anisotropy
.. doxygenfunction:: Hamiltonian_Get_Exchange_Shells
.. doxygenfunction:: Hamiltonian_Get_Exchange_N_Pairs
.. doxygenfunction:: Hamiltonian_Get_Exchange_Pairs
.. doxygenfunction:: Hamiltonian_Get_DMI_Shells
.. doxygenfunction:: Hamiltonian_Get_DMI_N_Pairs
.. doxygenfunction:: Hamiltonian_Get_DDI
.. doxygenfunction:: Hamiltonian_Write_Hessian
