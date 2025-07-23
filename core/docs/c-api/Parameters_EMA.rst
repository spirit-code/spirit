Parameters_EMA
=========================

.. code-block:: C

  #include "Spirit/Parameters_EMA.h"


This method, if needed, calculates modes (they can also be read in from a file)
and perturbs the spin system periodically in the direction of the eigenmode.


.. doxygenfunction:: Parameters_EMA_Clear_Modes

Set Parameters
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_EMA_Set_N_Modes
.. doxygenfunction:: Parameters_EMA_Set_N_Mode_Follow
.. doxygenfunction:: Parameters_EMA_Set_Frequency
.. doxygenfunction:: Parameters_EMA_Set_Amplitude
.. doxygenfunction:: Parameters_EMA_Set_Snapshot
.. doxygenfunction:: Parameters_EMA_Set_Sparse


Get Parameters
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_EMA_Get_N_Modes
.. doxygenfunction:: Parameters_EMA_Get_N_Mode_Follow
.. doxygenfunction:: Parameters_EMA_Get_Frequency
.. doxygenfunction:: Parameters_EMA_Get_Amplitude
.. doxygenfunction:: Parameters_EMA_Get_Snapshot
.. doxygenfunction:: Parameters_EMA_Get_Sparse
