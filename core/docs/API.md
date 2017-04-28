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

### State Managment

The State struct is passed around in an application to make the simulation's state available.

| State manipulation function                               | Effect |
| --------------------------------------------------------- | ------ |
| `State_Setup( const char * )`                             | Create new state by passing a config file |
| `State_Update( State * )`                                 | Update the state to hold current values |
| `State_Delete( State * )`                                 | Delete a state |
| `State_To_Config( State *, const char *, const char * )`  | Write a config file which will result in the same state if used in `State_Setup()`  |
| `State_DateTime( State * )`                               | Get datetime tag of the creation of the state |

A new state can be created with `State_Setup()`, where you can pass
a config file specifying your initial system parameters

```C
#import "Spirit/State.h"

State * p_state = State_Setup("");
```

If you do not pass a config file, the implemented defaults are used. *Note that you currently
cannot change the geometry of the systems in your state once they are
initialized.*

### System

| System Information           | Effect |
| ---------------------------- | ------------ |
| `System_Get_Index( State *)` | Returns System's Index |
| `System_Get_NOS( State *, int, int )` | Return System's number of spins |

| System Data                                             | Effect |
| -------------------------------------------------       | ------ |
| `System_Get_Spin_Directions( State *, int, int )`       | Get System's spin direction |
| `System_Get_Spin_Effective_Field( State *, int, int )`  | Get System's spin effective field |
| `System_Get_Rx( State *, int, int )`                    | Get a System's reaction coordinate in it's chain |
| `System_Get_Energy( State *, int, int )`                | Get System's energy |
| `System_Get_Energy_Array( State *, float *, int, int )` | Energy Array |

| System Output                                           | Effect |
| ------------------------------------------------------- | ------ |
| `System_Print_Energy_Array( State *, int, int )`        | Print on the console State's energy array |

| System Update                                           | Effect |
| ------------------------------------------------------- | ------ |
| `System_Update_Data( State *, int, int )`               | Update State's data. Used mainly for plotting |


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
