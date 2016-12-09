Core
---------

This is the core library of the Spirit framework.
It is meant to provide useful and easy API functions to enable productive work
with atomistic dynamics simulations and optimizations.
The current implementation is specific to atomistic spin models, but it may
easily be generalised.

### C interface

The API is exposed as a C interface and revolves around a simulation `State`
and may also be used from other languages, such as Python, Julia or even
JavaScript (see *ui-web*).

The `State` contains all the necessary data and keeps track of running
Optimizers. A new state can be created with `State_Setup`, where you can pass
a config file specifying your initial system parameters. If you do not pass a
config file, the implemented defaults are used. *Note that you currently
cannot change the geometry of the systems in your state once they are
initialized.*

The interface exposes functions for:
* Control of pimulations
* Manipulation of parameters
* Extracting information
* Generating spin configurations and transitions
* Logging messages
* Reading spin configurations
* Saving datafiles