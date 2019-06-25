UI - CPP
------------

By setting the CMake switch `SPIRIT_UI_CXX_USE_QT`, you can choose wether to use a regular
command line interface or a powerful QT user interface.

Both interfaces can be stopped with `Ctrl+C`.
Doing this will cause it to stop the current solver, write the corresponding output
files and the Log to your disk and terminate when finished.

Pressing `Ctrl+C` twice within 2 seconds will cause immediate termination.

### Command Line
This is a very simplistic command line user interface implementing the **core** library.
It has no non-standard library dependencies and should thus run almost anywhere,
where a compiler with C++11 support is available.

#### Controlling the code
The actions need to be hard-coded into `main.cpp`, there is currently no way to
script the actions of the code.



### QT UI
This is a QT5 user interface making use of the **core** and the **VFRendering**
libraries. It enables the user to control parameters during simulations and provides
a powerful live visualisation of the spin systems.

#### Widgets
The available calculation methods and solvers, starting and stopping the calculation
as well as the switching between images are implemented in `ControlWidget`. Via the file
menu, one can import and export e.g. spin configurations.
The `Spin Widget` wraps the **VFRendering** library's visualisation capabilities
and provides user interaction.
The remaining widgets are used to control parameters and view output. They are packed
into a `QDockWidget`, meaning they are repositionable and can be toggled. 

#### ui files
The *ui* folder contains QT-specific xml files, which can be edited using
*QT Creator* or *QT Designer*. 