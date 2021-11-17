

Simulation
====================================================================

```C
#include "Spirit/Simulation.h"
```

This API of Spirit is used to run and monitor iterative calculation methods.

If many iterations are called individually, one should use the single shot simulation functionality.
It avoids the allocations etc. involved when a simulation is started and ended and behaves like a
regular simulation, except that the iterations have to be triggered manually.



Definition of solvers
--------------------------------------------------------------------

Note that the VP and LBFGS Solvers are only meant for direct minimization and not for dynamics.



### Solver_VP

```C
Solver_VP          0
```

`VP`: Verlet-like velocity projection



### Solver_SIB

```C
Solver_SIB         1
```

`SIB`: Semi-implicit midpoint method B



### Solver_Depondt

```C
Solver_Depondt     2
```

`Depondt`: Heun method using rotations



### Solver_Heun

```C
Solver_Heun        3
```

`Heun`: second-order midpoint



### Solver_RungeKutta4

```C
Solver_RungeKutta4 4
```

`RK4`: 4th order Runge-Kutta



### Solver_LBFGS_OSO

```C
Solver_LBFGS_OSO   5
```

`LBFGS_OSO`: Limited memory Broyden-Fletcher-Goldfarb-Shanno, exponential transform



### Solver_LBFGS_Atlas

```C
Solver_LBFGS_Atlas 6
```

`LBFGS_Atlas`: Limited memory Broyden-Fletcher-Goldfarb-Shannon, stereographic projection



### Solver_VP_OSO

```C
Solver_VP_OSO      7
```

`Solver_VP_OSO`: Verlet-like velocity projection, exponential transform



Start or stop a simulation
--------------------------------------------------------------------



### PREFIX

```C

```

Monte Carlo



### Simulation_LLG_Start

```C
void Simulation_LLG_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1, bool singleshot=false, int idx_image=-1, int idx_chain=-1)
```

Landau-Lifshitz-Gilbert dynamics and energy minimisation



### Simulation_GNEB_Start

```C
void Simulation_GNEB_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1, bool singleshot=false, int idx_chain=-1)
```

Geodesic nudged elastic band method



### Simulation_MMF_Start

```C
void Simulation_MMF_Start(State *state, int solver_type, int n_iterations=-1, int n_iterations_log=-1, bool singleshot=false, int idx_image=-1, int idx_chain=-1)
```

Minimum mode following method



### Simulation_EMA_Start

```C
void Simulation_EMA_Start(State *state, int n_iterations=-1, int n_iterations_log=-1, bool singleshot=false, int idx_image=-1, int idx_chain=-1)
```

Eigenmode analysis



### Simulation_SingleShot

```C
void Simulation_SingleShot(State *state, int idx_image=-1, int idx_chain=-1)
```

Single iteration of a Method

If `singleshot=true` was passed to `Simulation_..._Start` before, this will perform one iteration.
Otherwise, nothing will happen.



### Simulation_N_Shot

```C
void Simulation_N_Shot(State *state, int N, int idx_image=-1, int idx_chain=-1)
```

N iterations of a Method

If `singleshot=true` was passed to `Simulation_..._Start` before, this will perform N iterations.
Otherwise, nothing will happen.



### Simulation_Stop

```C
void Simulation_Stop(State *state, int idx_image=-1, int idx_chain=-1)
```

Stop a simulation running on an image or chain



### Simulation_Stop_All

```C
void Simulation_Stop_All(State *state)
```

Stop all simulations



Get information
--------------------------------------------------------------------



### Simulation_Get_MaxTorqueComponent

```C
float Simulation_Get_MaxTorqueComponent(State * state, int idx_image=-1, int idx_chain=-1)
```

Get maximum torque component.

If a MC, LLG, MMF or EMA simulation is running this returns the max. torque on the current image.

If a GNEB simulation is running this returns the max. torque on the current chain.



### Simulation_Get_Chain_MaxTorqueComponents

```C
void Simulation_Get_Chain_MaxTorqueComponents(State * state, float * torques, int idx_chain=-1)
```

Get maximum torque components on the images of a chain.

Will only work if a GNEB simulation is running.



### Simulation_Get_MaxTorqueNorm

```C
float Simulation_Get_MaxTorqueNorm(State * state, int idx_image=-1, int idx_chain=-1)
```

Get maximum torque norm.

If a MC, LLG, MMF or EMA simulation is running this returns the max. torque on the current image.

If a GNEB simulation is running this returns the max. torque on the current chain.



### Simulation_Get_Chain_MaxTorqueNorms

```C
void Simulation_Get_Chain_MaxTorqueNorms(State * state, float * torques, int idx_chain=-1)
```

Get maximum torque norms on the images of a chain.

Will only work if a GNEB simulation is running.



### Simulation_Get_IterationsPerSecond

```C
float Simulation_Get_IterationsPerSecond(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the iterations per second (IPS).

If a MC, LLG, MMF or EMA simulation is running this returns the IPS on the current image.

If a GNEB simulation is running this returns the IPS on the current chain.



### Simulation_Get_Iteration

```C
int Simulation_Get_Iteration(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of iterations performed by the current simulation so far.



### Simulation_Get_Time

```C
float Simulation_Get_Time(State *state, int idx_image=-1, int idx_chain=-1)
```

Get time passed by the simulation [ps]

**Returns:**
- if an LLG simulation is running returns the cumulatively summed time steps `dt`
- otherwise returns 0



### Simulation_Get_Wall_Time

```C
int Simulation_Get_Wall_Time(State *state, int idx_image=-1, int idx_chain=-1)
```

Get number of miliseconds of wall time since the simulation was started



### Simulation_Get_Solver_Name

```C
const char * Simulation_Get_Solver_Name(State *state, int idx_image=-1, int idx_chain=-1)
```

Get name of the currently used method.

If a MC, LLG, MMF or EMA simulation is running this returns the Solver name on the current image.

If a GNEB simulation is running this returns the Solver name on the current chain.



### Simulation_Get_Method_Name

```C
const char * Simulation_Get_Method_Name(State *state, int idx_image=-1, int idx_chain=-1)
```

Get name of the currently used method.

If a MC, LLG, MMF or EMA simulation is running this returns the Method name on the current image.

If a GNEB simulation is running this returns the Method name on the current chain.



Whether a simulation is running
--------------------------------------------------------------------



### Simulation_Running_On_Image

```C
bool Simulation_Running_On_Image(State *state, int idx_image=-1, int idx_chain=-1)
```

Check if a simulation is running on specific image of specific chain



### Simulation_Running_On_Chain

```C
bool Simulation_Running_On_Chain(State *state, int idx_chain=-1)
```

Check if a simulation is running across a specific chain



### Simulation_Running_Anywhere_On_Chain

```C
bool Simulation_Running_Anywhere_On_Chain(State *state, int idx_chain=-1)
```

Check if a simulation is running on any or all images of a chain

