Core
---------

This is a library to work with atomistic dynamics and optimizations.
Its current implementation is specific to atomistic spin models, but it may easily be generalised.
The library will expose a *C-interface*, which revolves around a simulation `State` and may be used
from other Languages, such as JavaScript (see *ui-web*) or Python.

### C interface

**Note**: The interface is not yet implemented in this library.

The `State` contains a *Spin System Chain*, a *Solver* and an *Optimizer*:
    
    struct State {
      std::shared_ptr<Data::Spin_System_Chain> c;
      std::shared_ptr<Engine::Optimizer> optim;
      int active_image;
    };

The interface will expose the following functions:
* createSimulation
* getSpinDirections
* performIteration
* Configuration_DomainWall etc.
* Hamiltonian_Set_Boundary_Conditions etc. and corresponding 'Get' functions
* Parameters_Set_LLG_Time_Step etc. and corresponding 'Get' functions
* Geometry_Get_Bounds
* ...