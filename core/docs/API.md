SPIRIT API
====================

This will list the available API functions of the Spirit library.

The API is exposed as a C interface and may thus also be used from other
languages, such as Python, Julia or even JavaScript (see *ui-web*).
The API revolves around a simulation `State` which contains all the necessary
data and keeps track of running Solvers.

The API exposes functions for:
* Control of simulations
* Manipulation of parameters
* Extracting information
* Generating spin configurations and transitions
* Logging messages
* Reading spin configurations
* Saving datafiles


State Managment
---------------

To create a new state with one chain containing a single image,
initialized by an [input file](INPUT.md), and run the most simple example
of a **spin dynamics simulation**:
```C
#import "Spirit/State.h"
#import "Spirit/Simulation.h"

const char * cfgfile = "input/input.cfg";    // Input file
State * p_state = State_Setup(cfgfile);      // State setup
Simulation_PlayPause(p_state, "LLG", "SIB"); // Start a LLG simulation using the SIB solver
State_Delete(p_state)                        // State cleanup
```

A new state can be created with `State_Setup()`, where you can pass
a [config file](INPUT.md) specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
**Note that you currently cannot change the geometry of the systems
in your state once they are initialized.**

The State struct is passed around in an application to make the simulation's state available.

| State manipulation function                                                                                 | Return    | Effect |
| ----------------------------------------------------------------------------------------------------------- | --------- | ------ |
| `State_Setup( const char * config_file )`                                                                   | `State *` | Create new state by passing a config file |
| `State_Update( State * )`                                                                                   | `void`    | Update the state to hold current values |
| `State_Delete( State * )`                                                                                   | `void`    | Delete a state |
| `State_To_Config( State *, const char * config_file, const char * original_config_file)`                    | `void`    | Write a config file which will result in the same state if used in `State_Setup()`  |
| `State_DateTime( State * )`                                                                                 | `const char *` | Get datetime tag of the creation of the state |


System
------

| System Information                                              | Return           | Effect                          |
| --------------------------------------------------------------- | ---------------- | ------------------------------- |
| `System_Get_Index( State *)`                                    | `int`            | Returns System's Index          |
| `System_Get_NOS( State *, int idx_image, int idx_chain )`       | `int`            | Return System's number of spins |

| System Data                                                                                     | Return     | Effect |
| ------------------------------------------------------------------------------------------------| ---------- | ------ |
| `System_Get_Spin_Directions( State *, int idx_image, int idx_chain )`                           | `scalar *` | Get System's spin direction |
| `System_Get_Spin_Effective_Field( State *, int idx_image, int idx_chain )`                      | `scalar *` | Get System's spin effective field |
| `System_Get_Rx( State *, int idx_image, int idx_chain)`                                         | `float`    | Get a System's reaction coordinate in it's chain |
| `System_Get_Energy( State *, int idx_image, int idx_chain )`                                    | `float`    | Get System's energy |
| `System_Get_Energy_Array( State * energies, float * energies, int idx_image, int idx_chain )`   | `void`     | Energy Array (Should NOT be used) |

| System Output                                                              | Effect |
| -------------------------------------------------------------------------- | ------ |
| `System_Print_Energy_Array( State *, int idx_image, int idx_chain)`        | Print on the console State's energy array |

| System Update                                                              | Effect |
| -------------------------------------------------------------------------- | ------ |
| `System_Update_Data( State *, int idx_image, int idx_chain)`               | Update State's data. Used mainly for plotting |

Simulation
----------

With `Simulation_*` functions one can control and get information from the State's Solvers and their Methods.

| Simulation Basics                                                          | Effect |
| -------------------------------------------------------------------------- | ------ |
| `Simulation_SingleShot( State *, const char * c_method_type, const char * c_solver_type, int n_iterations, int n_iterations_log, int idx_image, int idx_chain )` | Executes a single Optimization iteration with a given method <br/> (CAUTION! does not check for already running simulations) |
| `Simulation_PlayPause( State *, const char * c_method_type, const char * c_solver_type, int n_iterations, int n_iterations_log, int idx_image, int idx_chain )` | Play/Pause functionality |
| `Simulation_Stop_All( State * )` | Stop all State's simulations |

| Simulation Data                                                                 | Return          | Effect |
| ------------------------------------------------------------------------------- | --------------- | ------ |
| `Simulation_Get_MaxTorqueComponent( State *, int idx_image, int idx_chain )`    | `float`         | Get Simulation's maximum torque component  |
| `Simulation_Get_IterationsPerSecond( State *, int idx_image, int idx_chain )`   | `float`         | Get Simulation's iterations per second     |
| `Simulation_Get_Solver_Name( State *, int idx_image, int idx_chain )`           | `const char *`  | Get Solver's name                       |
| `Simulation_Get_Method_Name( State *, int idx_image, int idx_chain )`           | `const char *`  | Get Method's name                          |

| Simulation Running Checking                                                     | Return          |
| ------------------------------------------------------------------------------- | --------------- |
| `Simulation_Running_Any_Anywhere( State * )`                                    | `bool`          |
| `Simulation_Running_LLG_Anywhere( State * )`                                    | `bool`          |
| `Simulation_Running_GNEB_Anywhere( State * )`                                   | `bool`          |
| `Simulation_Running_LLG_Chain( State *state, int idx_chain )`                   | `bool`          |
| `Simulation_Running_Any( State *, int idx_image, int idx_chain )`               | `bool`          |
| `Simulation_Running_LLG( State *, int idx_image, int idx_chain)`                | `bool`          |
| `Simulation_Running_GNEB( State *, int idx_chain )`                             | `bool`          |
| `Simulation_Running_MMF( State * )`                                             | `bool`          |

Geometry
--------

| Geometry Data                                                                                                      | Return        |
| ------------------------------------------------------------------------------------------------------------------ | ------------- |
| `Geometry_Get_Spin_Positions( State *, int idx_image, int idx_chain )`                                             | `scalar *`    |
| `Geometry_Get_Bounds( State *, float min[3], float max[3], int idx_image, int idx_chain )`                         | `void`        |
| `Geometry_Get_Center( State *, float center[3], int idx_image, int idx_chain )`                                    | `void`        |
| `Geometry_Get_Cell_Bounds( State *, float min[3], float max[3], int idx_image, int idx_chain )`                    | `void`        |
| `Geometry_Get_Basis_Vectors( State *, float a[3], float b[3], float c[3], int idx_image, int idx_chain )`          | `int`         |
| `Geometry_Get_N_Basis_Atoms( State *, int idx_image, int idx_chain )`                                              | `void`        |
| `Geometry_Get_N_Cells( State *, int n_cells[3], int idx_image, int idx_chain )`                                    | `void`        |
| `Geometry_Get_Translation_Vectors( State *, float ta[3], float tb[3], float tc[3], int idx_image, int idx_chain )` | `void`        |
| `Geometry_Get_Dimensionality( State *, int idx_image, int idx_chain )`                                             | `int`         |
| `Geometry_Get_Triangulation( State *, const int **indices_ptr, int n_cell_step, int idx_image, int idx_chain )`    | `int`         |

Transitions
-----------

| Transitions                                                                                                | Return           | Effect       |
| ---------------------------------------------------------------------------------------------------------- | ---------------- | ------------ |
| `Transition_Homogeneous( State *, int idx_1, int idx_2, int idx_chain=-1)`                                 | `void`           | -            |
| `Transition_Add_Noise_Temperature( State *, float temperature, int idx_1, int idx_2, int idx_chain)`       | `void`           | -            |

Quantities
----------

| Quantities                                                                                                    | Return           | Effect      |
| ------------------------------------------------------------------------------------------------------------- | ---------------- | ----------- |
| `Quantity_Get_Magnetization( State *, float m[3], int idx_image, int idx_chain )`                             | `void`           | -           |
| `Quantity_Get_Topological_Charge( State *, int idx_image, int idx_chain )`                                    | `float`          | -           |

Parameters
----------

| LLG Parameters Set                                                                                            | Return   | Effect      |
| ------------------------------------------------------------------------------------------------------------- | -------- | ----------- |
| `Parameters_Set_LLG_Time_Step( State *, float dt, int idx_image, int idx_chain )`                             | `void`   | -           |
| `Parameters_Set_LLG_Damping( State *, float damping, int idx_image, int idx_chain )`                          | `void`   | -           |
| `Parameters_Set_LLG_N_Iterations( State *, int n_iterations, int idx_image, int idx_chain )`                  | `void`   | -           |

| GNEB Parameters Set                                                                                           | Return   | Effect      |
| ------------------------------------------------------------------------------------------------------------- | -------- | ----------- |
| `Parameters_Set_GNEB_Spring_Constant( State *, float spring_constant, int idx_image, int idx_chain )`         | `void`   | -           |
| `Parameters_Set_GNEB_Climbing_Falling( State *, int image_type, int idx_image, int idx_chain )`               | `void`   | -           |
| `Parameters_Set_GNEB_N_Iterations( State *, int n_iterations, int idx_chain )`                                | `void`   | -           |

| LLG Parameters Get                                                                                            | Return   | Effect      |
| ------------------------------------------------------------------------------------------------------------- | -------- | ----------- |
| `Parameters_Get_LLG_Time_Step( State *, float * dt, int idx_image, int idx_chain)`                            | `void`   | -           |
| `Parameters_Get_LLG_Damping( State *, float * damping, int idx_image, int idx_chain)`                         | `void`   | -           |
| `Parameters_Get_LLG_N_Iterations( State *, int idx_image, int idx_chain)`                                     | `void`   | -           |

| GNEB Parameters Get                                                                                           | Return   | Effect      |
| ------------------------------------------------------------------------------------------------------------- | -------- | ----------- |
| `Parameters_Get_GNEB_Spring_Constant( State *, float * spring_constant, int idx_image, int idx_chain )`       | `void`   | -           |
| `Parameters_Get_GNEB_Climbing_Falling( State *, int * image_type, int idx_image, int idx_chain )`             | `void`   | -           |
| `Parameters_Get_GNEB_N_Iterations( State *, int idx_chain )`                                                  | `int`    | -           |
| `Parameters_Get_GNEB_N_Energy_Interpolations( State *, int idx_chain )`                                       | `int`    | -           |

Chain
-----

Get Chain's information

| Chain Info                                       | Return   | Effect |
| ------------------------------------------------ | -------- | ------ |
| `Chain_Get_Index( State * )`                     | `int`    | -      |
| `Chain_Get_NOI( State *, int idx_chain )`        | `int`    | -      |

Move between images (change `active_image` )

| Chain moving                                                      | Return  |
| ----------------------------------------------------------------- | ------- |
| `Chain_prev_Image( State *, int idx_chain )`                      | `bool`  |
| `Chain_next_Image( State *, int idx_chain )`                      | `bool`  |
| `Chain_Jump_To_Image( State *, int idx_image, int idx_chain )`    | `bool`  |

Insertion/deletion and replacement of images are done by

| Chain control                                                           | Return  |
| ----------------------------------------------------------------------- | ------- |
| `Chain_Image_to_Clipboard( State *, int idx_image, int idx_chain )`     | `void`  |
| `Chain_Replace_Image( State *, int idx_image, int idx_chain )`          | `void`  |
| `Chain_Insert_Image_Before( State *, int idx_image, int idx_chain )`    | `void`  |
| `Chain_Insert_Image_After( State *, int idx_image, int idx_chain )`     | `void`  |
| `Chain_Push_Back( State *, int idx_chain )`                             | `void`  |
| `Chain_Delete_Image( State *, int idx_image, int idx_chain )`           | `bool`  |
| `Chain_Pop_Back( State *, int idx_chain )`                              | `bool`  |

| Chain data                                                                           | Return  |
| ------------------------------------------------------------------------------------ | ------- |
| `Chain_Get_Rx( State *, float* Rx, int idx_chain )`                                  | `void`  |
| `Chain_Get_Rx_Interpolated( State *, float * Rx_interpolated, int idx_chain )`       | `void`  |
| `Chain_Get_Energy( State *, float* energy, int idx_chain )`                          | `void`  |
| `Chain_Get_Energy_Interpolated( State *, float* E_interpolated, int idx_chain )`     | `void`  |

| Chain data update                                   | Return  |
| --------------------------------------------------- | ------- |
| `Chain_Update_Data( State *, int idx_chain )`       | `void`  |
| `Chain_Setup_Data( State *, int idx_chain )`        | `void`  |


Hamiltonian
-----------

Set Hamiltonian's parameters

| Hamiltonian Set                                                                                               | Return   |
| ------------------------------------------------------------------------------------------------------------- | -------- |
| `Hamiltonian_Set_Boundary_Conditions( State *, const bool* periodical, int idx_image, int idx_chain )`        | `void`   |
| `Hamiltonian_Set_mu_s( State *, float mu_s, int idx_image, int idx_chain )`                                   | `void`   |
| `Hamiltonian_Set_Field( State *, float magnitude, const float* normal, int idx_image, int idx_chain )`        | `void`   |
| `Hamiltonian_Set_Exchange( State *, int n_shells, const float* jij, int idx_image, int idx_chain )`           | `void`   |
| `Hamiltonian_Set_DMI( State *, float dij, int idx_image, int idx_chain )`                                     | `void`   |
| `Hamiltonian_Set_BQE( State *, float dij, int idx_image, int idx_chain )`                                     | `void`   |
| `Hamiltonian_Set_FSC( State *, float dij, int idx_image, int idx_chain )`                                     | `void`   |
| `Hamiltonian_Set_Anisotropy( State *, float magnitude, const float* normal, int idx_image, int idx_chain )`   | `void`   |
| `Hamiltonian_Set_STT( State *, float magnitude, const float * normal, int idx_image, int idx_chain )`         | `void`   |
| `Hamiltonian_Set_Temperature( State *, float T, int idx_image, int idx_chain )`                               | `void`   |

Get Hamiltonian's parameters

| Hamiltonian Get                                                                                               | Return          |
| ------------------------------------------------------------------------------------------------------------- | --------------- |
| `Hamiltonian_Get_Name( State *, int idx_image, int idx_chain )`                                               | `const char *`  | 
| `Hamiltonian_Get_Boundary_Conditions( State *, bool* periodical, int idx_image, int idx_chain )`              | `void`          |
| `Hamiltonian_Get_mu_s( State *, float* mu_s, int idx_image, int idx_chain )`                                  | `void`          |
| `Hamiltonian_Get_Field( State *, float* magnitude, float* normal, int idx_image, int idx_chain )`             | `void`          |
| `Hamiltonian_Get_Exchange( State *, int* n_shells, float* jij, int idx_image, int idx_chain )`                | `void`          |
| `Hamiltonian_Get_Anisotropy( State *, float* magnitude, float* normal, int idx_image, int idx_chain )`        | `void`          |
| `Hamiltonian_Get_DMI( State *, float* dij, int idx_image, int idx_chain )`                                    | `void`          |
| `Hamiltonian_Get_BQE( State *, float* dij, int idx_image, int idx_chain )`                                    | `void`          |
| `Hamiltonian_Get_FSC( State *, float* dij, int idx_image, int idx_chain )`                                    | `void`          |
| `Hamiltonian_Get_STT( State *, float* magnitude, float* normal, int idx_image, int idx_chain )`               | `void`          |
| `Hamiltonian_Get_Temperature( State *, float* T, int idx_image, int idx_chain )`                              | `void`          |

Constants
---------

| Constants            | Return    | Description                        |
| -------------------- | --------- | ---------------------------------- |
| `Constants_mu_B()`   | `scalar`  | The Bohr Magneton [meV / T]        |
| `Constants_k_B()`    | `scalar`  | The Boltzmann constant [meV / K]   |

Log
---

| Log Utilities                                                                                          | Return   | Effect        |
| ------------------------------------------------------------------------------------------------------ | ---------| ------------- |
| `Log_Send( State *, int level, int sender, const char * message, int idx_image, int idx_chain )`       | `void`   | Send a Log message |
| `Log_Get_Entries( State * )`                                                                           | `std::vector<Utility::LogEntry>`  | Get the entries from the Log and write new number of entries into given int |
| `Log_Append( State * )`                                                                                | `void`   | Append the Log to it's file            |
| `Log_Dump( State * )`                                                                                  | `void`   | Dump the Log into it's file            |
| `Log_Get_N_Entries( State * )`                                                                         | `int`    | Get the number of Log entries          |
| `Log_Get_N_Errors( State * )`                                                                          | `int`    | Get the number of errors in the Log    |
| `Log_Get_N_Warnings( State * )`                                                                        | `int`    | Get the number of warnings in the Log  |

Log macro variables for Levels 

| Log Levels            | value |
| --------------------- | :---: |
| `Log_Level_All`       | 0     |
| `Log_Level_Severe`    | 1     |
| `Log_Level_Error`     | 2     |
| `Log_Level_Warning`   | 3     |
| `Log_Level_Parameter` | 4     |
| `Log_Level_Info`      | 5     |
| `Log_Level_Debug`     | 6     |

Log macro variables for Senders 

| Log Senders        | value |
| ------------------ | :---: |
| `Log_Sender_All`   | 0     |
| `Log_Sender_IO`    | 1     |
| `Log_Sender_GNEB`  | 2     |
| `Log_Sender_LLG`   | 3     |
| `Log_Sender_MC`    | 4     |
| `Log_Sender_MMF`   | 5     |
| `Log_Sender_API`   | 6     |
| `Log_Sender_UI`    | 7     |

IO
--

| Macros of File Formats for Vector Fields | values  | Description                                       |
| ---------------------------------------- | :-----: | --------------------------------------------------|
| `IO_Fileformat_Regular`                  | 0       | sx sy sz (separated by whitespace)                |
| `IO_Fileformat_Regular_Pos`              | 1       | px py pz sx sy sz (separated by whitespace)       |
| `IO_Fileformat_CSV`                      | 2       | sx, sy, sz (separated by commas)                  |
| `IO_Fileformat_CSV_Pos`                  | 3       | px, py, pz, sx, sy, (sz separated by commas)      |
| `IO_Fileformat_OVF_bin8`                 | 4       | [OOMMF vector field (OVF) v2.0](http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html) file format |
| `IO_Fileformat_OVF_text`                 | 6       |                                                   |

Read and Write functions

| Images IO                                                                                                                 | Return     |
| ------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `IO_N_Images_In_File( State *state, const char *file, int format, int idx_chain )`                                        | `int`      |
| `IO_Image_Write( State *state, const char *file, int format, const char *comment, int idx_image, int idx_chain )`         | `void`     |
| `IO_Image_Append( State *state, const char *file, int format, const char *comment, int idx_image, int idx_chain )`        | `void`     |
| `IO_Image_Read( State *state, const char *file, int format, int idx_image_infile, int idx_image_inchain, int idx_chain )` | `void`     |

| Chains IO                                                                                                                          | Return     |
| ---------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `IO_Chain_Write( State *state, const char *file, int format,  const char *comment, int idx_chain )`                                | `void`     |
| `IO_Chain_Append( State *state, const char *file, int format, const char *comment, int idx_chain )`                                | `void`     |
| `IO_Chain_Read( State *state, const char *file, int format, int starting_image, int ending_image, int insert_idx, int idx_chain )` | `void`     |

Energies from `System` and `Chain`

| System Energies                                                                                | Return     |
| ---------------------------------------------------------------------------------------------- | ---------- |
| `IO_Image_Write_Energy_per_Spin( State *, const char* file, int idx_image, int idx_chain )`    | `void`     |
| `IO_Image_Write_Energy( State *, const char* file, int idx_image, int idx_chain )`             | `void`     |


| Chain Energies                                                                                 | Return     |
| ---------------------------------------------------------------------------------------------- | ---------- |
| `IO_Chain_Write_Energies( State *, const char* file, int idx_chain )`                          | `void`     |
| `IO_Chain_Write_Energies_Interpolated( State *, const char* file, int idx_chain )`             | `void`     |


---

[Home](Readme.md)