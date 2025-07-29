Parameters_MC
=========================

.. code-block:: C

  #include "Spirit/Parameters_MC.h"

.. doxygendefine:: MC_Metropolis_Step_Spin_Sphere 0
.. doxygendefine:: MC_Metropolis_Step_Spin_Cone 1
.. doxygendefine:: MC_Metropolis_Step_Spin_Semi_Classical 2

.. doxygenstruct:: Parameters_MC_Metropolis_Parameters
   :members:


Set Output
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_MC_Set_Output_Tag
.. doxygenfunction:: Parameters_MC_Set_Output_Folder
.. doxygenfunction:: Parameters_MC_Set_Output_General
.. doxygenfunction:: Parameters_MC_Set_Output_Energy
.. doxygenfunction:: Parameters_MC_Set_Output_Configuration


Set Parameters
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_MC_Set_N_Iterations
.. doxygenfunction:: Parameters_MC_Set_Temperature
.. doxygenfunction:: Parameters_MC_Set_Metropolis_Parameters
.. doxygenfunction:: Parameters_MC_Set_Metropolis_Cone
.. doxygenfunction:: Parameters_MC_Set_Random_Sample

Get Output
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_MC_Get_Output_Tag
.. doxygenfunction:: Parameters_MC_Get_Output_Folder
.. doxygenfunction:: Parameters_MC_Get_Output_General
.. doxygenfunction:: Parameters_MC_Get_Output_Energy
.. doxygenfunction:: Parameters_MC_Get_Output_Configuration


Get Parameters
--------------------------------------------------------------------

.. doxygenfunction:: Parameters_MC_Get_N_Iterations
.. doxygenfunction:: Parameters_MC_Get_Temperature
.. doxygenfunction:: Parameters_MC_Get_Metropolis_Cone
.. doxygenfunction:: Parameters_MC_Get_Metropolis_Parameters
.. doxygenfunction:: Parameters_MC_Get_Random_Sample
