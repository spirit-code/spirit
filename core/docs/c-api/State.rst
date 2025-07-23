State
=========================

.. code-block:: C

  #include "Spirit/State.h"

To create a new state with one chain containing a single image,
initialized by a :doc:`config file </core/docs/Input>`, and run
the most simple example of a **spin dynamics simulation**:

.. code-block:: C

  #import "Spirit/Simulation.h"
  #import "Spirit/State.h"

  const char * cfgfile = "input/input.cfg";  // Input file
  State * p_state = State_Setup(cfgfile);    // State setup
  Simulation_LLG_Start(p_state, Solver_SIB); // Start a LLG simulation using the SIB solver
  State_Delete(p_state)                      // State cleanup


.. doxygenstruct:: State

.. doxygenfunction:: State_Setup
.. doxygenfunction:: State_Delete
.. doxygenfunction:: State_Update
.. doxygenfunction:: State_To_Config
.. doxygenfunction:: State_DateTime
