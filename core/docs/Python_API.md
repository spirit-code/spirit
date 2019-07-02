Usage
====================================================


Energy minimisation
----------------------------------------------------

Energy minimisation of a spin system can be performed
using the LLG method and the velocity projection (VP)
solver:

```Python
from spirit import simulation, state

with state.State("input/input.cfg") as p_state:
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP)
```

or using one of the dynamics solvers, using dissipative
dynamics:

```Python
from spirit import parameters, simulation, state

with state.State("input/input.cfg") as p_state:
    parameters.llg.set_direct_minimization(p_state, True)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT)
```


LLG method
----------------------------------------------------

To perform an LLG dynamics simulation:

```Python
from spirit import simulation, state

with state.State("input/input.cfg") as p_state:
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT)
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


```Python
from spirit import state, chain, configuration, transition, simulation 

noi = 7

with state.State("input/input.cfg") as p_state:
    ### Copy the first image and set chain length
    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, noi)

    ### First image is homogeneous with a Skyrmion in the center
    configuration.plus_z(p_state, idx_image=0)
    configuration.skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=0)
    ### Last image is homogeneous
    configuration.plus_z(p_state, idx_image=noi-1)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=noi-1)

    ### Create initial guess for transition: homogeneous rotation
    transition.homogeneous(p_state, 0, noi-1)

    ### Initial GNEB relaxation
    simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP, n_iterations=5000)
    ### Automatically set climbing and falling images
    chain.set_image_type_automatically(p_state)
    ### Full GNEB relaxation
    simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP)
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