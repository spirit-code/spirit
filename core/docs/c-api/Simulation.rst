Simulation
=========================

.. code-block:: C

  #include "Spirit/Simulation.h"

This API of Spirit is used to run and monitor iterative calculation methods.

If many iterations are called individually, one should use the single shot simulation functionality.
It avoids the allocations etc. involved when a simulation is started and ended and behaves like a
regular simulation, except that the iterations have to be triggered manually.

Definition of solvers
--------------------------------------------------------------------

Note that the VP and LBFGS Solvers only do direct minimization and not suitable for dynamics.

.. doxygendefine:: Solver_VP
.. doxygendefine:: Solver_SIB
.. doxygendefine:: Solver_Depondt
.. doxygendefine:: Solver_Heun
.. doxygendefine:: Solver_RungeKutta4
.. doxygendefine:: Solver_LBFGS_OSO
.. doxygendefine:: Solver_LBFGS_Atlas
.. doxygendefine:: Solver_VP_OSO

Definition of Monte Carlo Algorithms
--------------------------------------------------------------------

.. doxygendefine:: MC_Algorithm_Metropolis
.. doxygendefine:: MC_Algorithm_Metropolis_MDC

Run info
--------------------------------------------------------------------

.. doxygenstruct:: Simulation_Run_Info
   :members:

.. doxygenfunction:: free_run_info

Start or stop a simulation
--------------------------------------------------------------------

.. doxygenfunction:: Simulation_MC_Start
.. doxygenfunction:: Simulation_LLG_Start
.. doxygenfunction:: Simulation_GNEB_Start
.. doxygenfunction:: Simulation_MMF_Start
.. doxygenfunction:: Simulation_EMA_Start
.. doxygenfunction:: Simulation_SingleShot
.. doxygenfunction:: Simulation_N_Shot
.. doxygenfunction:: Simulation_Stop
.. doxygenfunction:: Simulation_Stop_All


Get information
--------------------------------------------------------------------

.. doxygenfunction:: Simulation_Get_MaxTorqueNorm
.. doxygenfunction:: Simulation_Get_Chain_MaxTorqueNorms
.. doxygenfunction:: Simulation_Get_IterationsPerSecond
.. doxygenfunction:: Simulation_Get_Iteration
.. doxygenfunction:: Simulation_Get_Time
.. doxygenfunction:: Simulation_Get_Wall_Time


Whether a simulation is running
--------------------------------------------------------------------

.. doxygenfunction:: Simulation_Running_On_Image
.. doxygenfunction:: Simulation_Running_On_Chain
.. doxygenfunction:: Simulation_Running_Anywhere_On_Chain
