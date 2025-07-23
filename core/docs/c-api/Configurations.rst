Configurations
=========================

.. code-block:: C

  #include "Spirit/Configurations.h"

Setting spin configurations for individual spin systems.

The position of the relative center and a set of conditions can be defined.


Clipboard
--------------------------------------------------------------------

.. doxygenfunction:: Configuration_To_Clipboard
.. doxygenfunction:: Configuration_From_Clipboard
.. doxygenfunction:: Configuration_From_Clipboard_Shift


Nonlocalised
--------------------------------------------------------------------

.. doxygenfunction:: Configuration_Domain
.. doxygenfunction:: Configuration_PlusZ
.. doxygenfunction:: Configuration_MinusZ
.. doxygenfunction:: Configuration_Random
.. doxygenfunction:: Configuration_SpinSpiral
.. doxygenfunction:: Configuration_SpinSpiral_2q


Perturbations
--------------------------------------------------------------------

.. doxygenfunction:: Configuration_Add_Noise_Temperature
.. doxygenfunction:: Configuration_Displace_Eigenmode


Localised
--------------------------------------------------------------------

.. doxygenfunction:: Configuration_Skyrmion
.. doxygenfunction:: Configuration_DW_Skyrmion
.. doxygenfunction:: Configuration_Hopfion


Pinning and atom types
--------------------------------------------------------------------

This API can also be used to change the `pinned` state and the `atom type`
of atoms in a spacial region, instead of using translation indices.

.. doxygenfunction:: Configuration_Set_Pinned
.. doxygenfunction:: Configuration_Set_Atom_Type


Default Values
--------------------------------------------------------------------

.. doxygenvariable:: defaultPos
.. doxygenvariable:: defaultRect
.. doxygenvariable:: defaultNormal
