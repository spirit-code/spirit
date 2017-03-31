Spirit Python Library
---------------------

This library is meant to provide useful and easy API functions to enable productive work
with atomistic dynamics simulations and optimizations.
The current implementation is specific to atomistic spin models, but it may
easily be generalised.

The library is written in C++ but has been wrapped in Python for easier install and use.


### API functions

The API revolves around a simulation `State` which contains all the necessary
data and keeps track of running Optimizers.
A new `State` can be created with 

    from Spirit import state
    with state.State(cfgfile) as p_state:
        pass

where you can pass a config file specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
*Note that you currently cannot change the geometry of the systems in your state once they are initialized.*

The interface exposes functions for:
* Control of simulations
* Manipulation of parameters
* Extracting information
* Generating spin configurations and transitions
* Logging messages
* Reading spin configurations
* Saving datafiles


### Running a Simulation

An easy example is a Landau-Lifshitz-Gilbert (LLG) dynamics simulation
using the Semi-implicit method B (SIB):

    from spirit import state
    with state.State(cfgfile) as p_state:
        simulation.PlayPause(p_state, "LLG", "SIB")