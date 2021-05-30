#pragma once
#ifndef INTERFACE_SIMULATION_H
#define INTERFACE_SIMULATION_H
#include "DLL_Define_Export.h"
struct State;

#include <vector>

/*
Simulation
====================================================================

```C
#include "Spirit/Simulation.h"
```

This API of Spirit is used to run and monitor iterative calculation methods.

If many iterations are called individually, one should use the single shot simulation functionality.
It avoids the allocations etc. involved when a simulation is started and ended and behaves like a
regular simulation, except that the iterations have to be triggered manually.
*/

/*
Definition of solvers
--------------------------------------------------------------------

Note that the VP and LBFGS Solvers are only meant for direct minimization and not for dynamics.
*/

// `VP`: Verlet-like velocity projection
#define Solver_VP          0

// `SIB`: Semi-implicit midpoint method B
#define Solver_SIB         1

// `Depondt`: Heun method using rotations
#define Solver_Depondt     2

// `Heun`: second-order midpoint
#define Solver_Heun        3

// `RK4`: 4th order Runge-Kutta
#define Solver_RungeKutta4 4

// `LBFGS_OSO`: Limited memory Broyden-Fletcher-Goldfarb-Shanno, exponential transform
#define Solver_LBFGS_OSO   5

// `LBFGS_Atlas`: Limited memory Broyden-Fletcher-Goldfarb-Shannon, stereographic projection
#define Solver_LBFGS_Atlas 6

// `Solver_VP_OSO`: Verlet-like velocity projection, exponential transform
#define Solver_VP_OSO      7

/*
Start or stop a simulation
--------------------------------------------------------------------
*/

// Monte Carlo
PREFIX void Simulation_MC_Start(State *state, int n_iterations=-1, int n_iterations_log=-1,
    bool singleshot=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Landau-Lifshitz-Gilbert dynamics and energy minimisation
PREFIX void Simulation_LLG_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1,
    bool singleshot=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Geodesic nudged elastic band method
PREFIX void Simulation_GNEB_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1,
    bool singleshot=false, int idx_chain=-1) SUFFIX;

// Minimum mode following method
PREFIX void Simulation_MMF_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1,
    bool singleshot=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Eigenmode analysis
PREFIX void Simulation_EMA_Start(State *state, int n_iterations=-1, int n_iterations_log=-1,
    bool singleshot=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Single iteration of a Method

If `singleshot=true` was passed to `Simulation_..._Start` before, this will perform one iteration.
Otherwise, nothing will happen.
*/
PREFIX void Simulation_SingleShot(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
N iterations of a Method

If `singleshot=true` was passed to `Simulation_..._Start` before, this will perform N iterations.
Otherwise, nothing will happen.
*/
PREFIX void Simulation_N_Shot(State *state, int N, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Stop a simulation running on an image or chain
PREFIX void Simulation_Stop(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Stop all simulations
PREFIX void Simulation_Stop_All(State *state) SUFFIX;

/*
Get information
--------------------------------------------------------------------
*/

/*
Get maximum torque component.

If a MC, LLG, MMF or EMA simulation is running this returns the max. torque on the current image.

If a GNEB simulation is running this returns the max. torque on the current chain.
*/
PREFIX float Simulation_Get_MaxTorqueComponent(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Get maximum torque components on the images of a chain.

Will only work if a GNEB simulation is running.
*/
PREFIX void Simulation_Get_Chain_MaxTorqueComponents(State * state, float * torques, int idx_chain=-1) SUFFIX;

/*
Get maximum torque norm.

If a MC, LLG, MMF or EMA simulation is running this returns the max. torque on the current image.

If a GNEB simulation is running this returns the max. torque on the current chain.
*/
PREFIX float Simulation_Get_MaxTorqueNorm(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Get maximum torque norms on the images of a chain.

Will only work if a GNEB simulation is running.
*/
PREFIX void Simulation_Get_Chain_MaxTorqueNorms(State * state, float * torques, int idx_chain=-1) SUFFIX;

/*
Returns the iterations per second (IPS).

If a MC, LLG, MMF or EMA simulation is running this returns the IPS on the current image.

If a GNEB simulation is running this returns the IPS on the current chain.
*/
PREFIX float Simulation_Get_IterationsPerSecond(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Returns the number of iterations performed by the current simulation so far.
PREFIX int Simulation_Get_Iteration(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Get time passed by the simulation [ps]

**Returns:**
- if an LLG simulation is running returns the cumulatively summed time steps `dt`
- otherwise returns 0
*/
PREFIX float Simulation_Get_Time(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Get number of miliseconds of wall time since the simulation was started
PREFIX int Simulation_Get_Wall_Time(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Get name of the currently used method.

If a MC, LLG, MMF or EMA simulation is running this returns the Solver name on the current image.

If a GNEB simulation is running this returns the Solver name on the current chain.
*/
PREFIX const char * Simulation_Get_Solver_Name(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Get name of the currently used method.

If a MC, LLG, MMF or EMA simulation is running this returns the Method name on the current image.

If a GNEB simulation is running this returns the Method name on the current chain.
*/
PREFIX const char * Simulation_Get_Method_Name(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Whether a simulation is running
--------------------------------------------------------------------
*/

// Check if a simulation is running on specific image of specific chain
PREFIX bool Simulation_Running_On_Image(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Check if a simulation is running across a specific chain
PREFIX bool Simulation_Running_On_Chain(State *state, int idx_chain=-1) SUFFIX;

// Check if a simulation is running on any or all images of a chain
PREFIX bool Simulation_Running_Anywhere_On_Chain(State *state, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif
