Usage
====================================================


Energy minimisation
----------------------------------------------------

Energy minimisation of a spin system can be performed
using the LLG method and the velocity projection (VP)
solver:

```C++
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <memory>

auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
Simulation_LLG_Start(state.get(), Solver_VP);
```

or using one of the dynamics solvers, using dissipative
dynamics:

```C++
#include <Spirit/Parameters.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <memory>

auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
Parameters_LLG_Set_Direct_Minimization(state.get(), true);
Simulation_LLG_Start(state.get(), Solver_Depondt);
```


LLG method
----------------------------------------------------

To perform an LLG dynamics simulation:

```C++
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <memory>

auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
Simulation_LLG_Start(state.get(), Solver_Depondt);
```

Note that the velocity projection (VP) solver is not a dynamics solver.


GNEB method
----------------------------------------------------

The geodesic nudged elastic band method.
See also the [method paper](http://www.sciencedirect.com/science/article/pii/S0010465515002696).

This method operates on a transition between two spin configurations,
discretised by "images" on a "chain". The procedure follows these steps:
1. set the number of images
2. set the initial and final spin configuration
3. create an initial guess for the transition path
4. run an initial GNEB relaxation
5. determine and set the suitable images on the chain to converge on extrema
6. run a full GNEB relaxation using climbing and falling images

```C++
#include <Spirit/Chain.h>
#include <Spirit/Configuration.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/Transition.h>
#include <memory>

int NOI = 7;

auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);

// Copy the first image and set chain length
Chain_Image_to_Clipboard(state.get());
Chain_Set_Length(state.get(), NOI);

// First image is homogeneous with a Skyrmion in the center
Configuration_Plus_Z(state.get(), 0);
Configuration_Skyrmion(state.get(), 5.0, phase=-90.0);
Simulation_LLG_Start(state.get(), Solver_VP);
// Last image is homogeneous
Configuration_Plus_Z(state.get(), NOI-1);
Simulation_LLG_Start(state.get(), simulation.SOLVER_VP, NOI-1);

// Create initial guess for transition: homogeneous rotation
Transition_Homogeneous(state.get(), 0, noi-1);

// Initial GNEB relaxation
Simulation_GNEB_Start(state.get(), Solver_VP, 5000);
// Automatically set climbing and falling images
Chain_Set_Image_Type_Automatically(state.get());
// Full GNEB relaxation
Simulation_GNEB_Start(state.get(), Solver_VP);
```


HTST
----------------------------------------------------

The harmonic transition state theory.
See also the [method paper](https://link.aps.org/doi/10.1103/PhysRevB.85.184409).

*The usage of this method is not yet documented.*


MMF method
----------------------------------------------------

The minimum mode following method.
See also the [method paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.197202).

*The usage of this method is not yet documented.*