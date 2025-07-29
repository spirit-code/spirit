System
=========================

.. code-block:: C

  #include "Spirit/System.h"

Spin systems are often referred to as "images" throughout Spirit.
The `idx_image` is used throughout the API to specify which system
out of the chain a function should be applied to.
`idx_image=-1` refers to the active image of the chain.


.. doxygenfunction:: System_Get_Index
.. doxygenfunction:: System_Get_NOS
.. doxygenfunction:: System_Get_Spin_Directions
.. doxygenfunction:: System_Get_Effective_Field
.. doxygenfunction:: System_Get_Eigenmode
.. doxygenfunction:: System_Get_Rx
.. doxygenfunction:: System_Get_Energy
.. doxygenfunction:: System_Get_Energy_Array_Names
.. doxygenfunction:: System_Get_Energy_Array
.. doxygenfunction:: System_Get_Eigenvalues
.. doxygenfunction:: System_Print_Energy_Array
.. doxygenfunction:: System_Update_Energy
.. doxygenfunction:: System_Update_Magnetization
.. doxygenfunction:: System_Update_Effective_Field
.. doxygenfunction:: System_Update_Data
.. doxygenfunction:: System_Update_Eigenmodes
