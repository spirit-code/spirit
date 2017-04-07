SPIRIT API
====================

This will list the available API functions of the Spirit library.

The API is exposed as a C interface and may thus also be used from other
languages, such as Python, Julia or even JavaScript (see *ui-web*).
The API revolves around a simulation `State` which contains all the necessary
data and keeps track of running Optimizers.

The API exposes functions for:
* Control of simulations
* Manipulation of parameters
* Extracting information
* Generating spin configurations and transitions
* Logging messages
* Reading spin configurations
* Saving datafiles



C API
----------
A new state can be created with `State_Setup`, where you can pass
a config file specifying your initial system parameters

```C
    #import "Spirit/State.h"

    State * p_state = State_Setup("");
```

If you do not pass a config file, the implemented defaults are used. *Note that you currently
cannot change the geometry of the systems in your state once they are
initialized.*



Python API
----------

A new `State` can be created with 

```python
    from spirit import state
    with state.State("") as p_state:
        pass
```

where you can pass a config file specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
*Note that you currently cannot change the geometry of the systems in your state once they are initialized.*
