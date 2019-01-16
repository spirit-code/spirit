

State
====================================================================

```C
#include "Spirit/Transitions.h"
```

To create a new state with one chain containing a single image,
initialized by an [input file](INPUT.md), and run the most simple example
of a **spin dynamics simulation**:

```C
#import "Spirit/State.h"
#import "Spirit/Simulation.h"

const char * cfgfile = "input/input.cfg";  // Input file
State * p_state = State_Setup(cfgfile);    // State setup
Simulation_LLG_Start(p_state, Solver_SIB); // Start a LLG simulation using the SIB solver
State_Delete(p_state)                      // State cleanup
```



### State

```C
struct State
```

The opaque state struct, containing all calculation data.

```
+-----------------------------+
| State                       |
| +-------------------------+ |
| | Chain                   | |
| | +--------------------+  | |
| | | 0th System ≡ Image |  | |
| | +--------------------+  | |
| | +--------------------+  | |
| | | 1st System ≡ Image |  | |
| | +--------------------+  | |
| |   .                     | |
| |   .                     | |
| |   .                     | |
| | +--------------------+  | |
| | | Nth System ≡ Image |  | |
| | +--------------------+  | |
| +-------------------------+ |
+-----------------------------+
```

This is passed to and is operated on by the API functions.

A new state can be created with `State_Setup()`, where you can pass
a [config file](INPUT.md) specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
**Note that you currently cannot change the geometry of the systems
in your state once they are initialized.**



### State_Setup

```C
State * State_Setup(const char * config_file = "", bool quiet = false)
```

Create the State and fill it with initial data.

- `config_file`: if a config file is given, it will be parsed for
  keywords specifying the initial values. Otherwise, defaults are used
- `quiet`: if `true`, the defaults are changed such that only very few
  messages will be printed to the console and no output files are written



### State_Delete

```C
void State_Delete(State * state)
```

Correctly deletes a State and frees the corresponding memory.



### State_Update

```C
void State_Update(State * state)
```

Update the state to hold current values.



### State_To_Config

```C
void State_To_Config(State * state, const char * config_file, const char * original_config_file="")
```

Write a config file which should give the same state again when
used in State_Setup (modulo the number of chains and images)



### State_DateTime

```C
const char * State_DateTime(State * state)
```

Returns a string containing the datetime tag (timepoint of creation) of this state.

Format: `yyyy-mm-dd_hh-mm-ss`

