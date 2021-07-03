Spirit immediate mode desktop UI
================================


![Logo](https://imgur.com/lGZNdop.png "Spirit Logo")

The cross-platform imgui desktop user interface provides an alternative to the
QT GUI and will eventually replace it.

It can be built by setting the CMake option `SPIRIT_UI_USE_IMGUI=ON`, see also
the build instructions for [Unix/OSX](Build_Unix_OSX.md) and [Windows](Build_Windows.md)
for information on how to build the graphical user interface on your machine.

_Note: This GUI is not yet fully fledged and should be regarded as a "preview"_

*Known Issues:*
- Energies in energy plot are not updating correctly


Physics features
----------------
The pysics features are on par with the QT GUI.

Insert Configurations:
- White noise
- (Anti-) Skyrmions
- Domains
- Spin Spirals

You can manipulate the Hamiltonian as well as simulation parameters and your
output file configuration.

You can start and stop simulation and directly interact with a running simulation.
- LLG Simulation: Dynamics and Minimization
- GNEB: create transitions and calculate minimum energy paths

By copying and inserting spin systems and manipulating them you may create
arbitrary transitions between different spin states to use them in GNEB calculations.
Furthermore you can choose different images to be climbing or falling images during
your calculation.


Real-time visualisation
-----------------------
The imgui provides the same visualisation capabilities as the QT GUI:

- Arrows, Surface (2D/3D), Isosurfaces
- Spins or eff. field
- Every n'th arrow
- Spin sphere
- Directional & position filters
- Various colourmaps

Additionally, it provides
- Sphere and dot renderers
- Individual control over each renderer via a separate widget

![Visualisation Settings](https://i.imgur.com/JWuqhKk.png "Visualisation Settings")


Additional features
-------------------
- Drag mode: drag, copy, insert, change radius
- Screenshot
- Read configuration or chain
- Save configuration or chain
- Take a screenshot


How to perform an LLG dynamics calculation
--------------------------------------------

To perform a dynamics simulation, use the "LLG" method and one of the following solvers:

- Depondt
- SIB
- Heun
- RK4

In this case, parameters such as temperature or spin
current will have an effect and the passed time has physical
meaning:

![LLG](https://i.imgur.com/j9bHhXb.png "LLG")

How to perform an energy minimisation
--------------------------------------------

The most straightforward way of minimising the energy of a
spin configuration is to use the "Minimizer" method and one of the following solvers:

- VP
- VP_OSO
- LBFGS_Atlas
- LBFGS_OSO

![Minimizer](https://i.imgur.com/GN4jc5E.png "Minimizer")

By pressing the "play" button or the space bar, the calculation is started.

How to perform a GNEB calculation
--------------------------------------------

Select the GNEB method and one of the minimisation solvers.

In order to perform a geodesic nudged elastic band (GNEB)
calculation, you need to first create a chain of spin systems,
in this context called "images".
You can set the chain length directly in the field nex to the start/stop button.

You can also manipulate the chain by copying the current image using `ctrl+c`
(on mac replace `ctrl` with `cmd`) and then `ctrl+rightarrow`/`ctrl+leftarrow`
to insert the copy into the chain and `ctrl+v` to overwrite the current image.
See the help menu for all keybindings.

The GUI will show the length of the chain:

![GUI controls](https://i.imgur.com/LDTSkwC.png "GUI controls")

You can use the buttons or the right and left arrow keys to
switch between images.

A data plot is available to visualise your chain of spin systems.
The interpolated energies become available when you run a GNEB
calculation.

Key bindings
------------

<i>Note that some of the keybindings may only work correctly on US keyboard layout.</i>

### UI Controls

| Effect                                                            | Keystroke                                            |
| ----------------------------------------------------------------- | :--------------------------------------------------: |
| Show this                                                         | <kbd>F1</kbd>                                        |
| Toggle Settings                                                   | <kbd>F2</kbd>                                        |
| Toggle Plots                                                      | <kbd>F3</kbd>                                        |
| Toggle Debug                                                      | <kbd>F4</kbd>                                        |
| Toggle \"Dragging\" mode                                          | <kbd>F5</kbd>                                        |
| Toggle large visualization                                        | <kbd>F10</kbd> / <kbd>Ctrl</kbd>+<kbd>F</kbd>        |
| Toggle full-screen window                                         | <kbd>F11</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>F</kbd> |
| Screenshot of Visualization region                                | <kbd>F12</kbd> / <kbd>Home</kbd>                     |
| Toggle OpenGL Visualization                                       | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>V</kbd>        |
| Try to return focus to main UI                                    | <kbd>Esc</kbd>                                       |

### Camera Controls

| Effect                                  | Keystroke                                                                                                   |
| --------------------------------------- | :---------------------------------------------------------------------------------------------------------: |
| Rotate the camera around                | <kbd>Left mouse</kbd> / <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> ( <kbd>Shift</kbd> to go slow)  |
| Move the camera around                  | <kbd>Left mouse</kbd> / <kbd>T</kbd> <kbd>F</kbd> <kbd>G</kbd> <kbd>H</kbd> ( <kbd>Shift</kbd> to go slow)  |
| Zoom in on focus point                  | <kbd>Scroll mouse</kbd> ( <kbd>Shift</kbd> to go slow)                                                      |
| Set the camera in X, Y or Z direction   | <kbd>X</kbd> <kbd>Y</kbd> <kbd>Z</kbd> ( <kbd>shift</kbd> to invert)                                        |

### Control Simulations

| Effect                                 | Keystroke                           |
| -------------------------------------- | :---------------------------------: |
| Play/Pause                             | <kbd>Space</kbd>                    |
| Cycle Method                           | <kbd>Ctrl</kbd>+<kbd>M</kbd>        |
| Cycle Solver                           | <kbd>Ctrl</kbd>+<kbd>S</kbd>        |

### Manipulate the current image

| Effect                                 | Keystroke                           |
| -------------------------------------- | :---------------------------------: |
| Random configuration                   | <kbd>Ctrl</kbd>+<kbd>R</kbd>        |
| Add tempered noise                     | <kbd>Ctrl</kbd>+<kbd>N</kbd>        |
| Insert last used configuration         | <kbd>Enter</kbd>                    |

### Visualisation

| Effect                                           | Keystroke                                   |
| ------------------------------------------------ | :-----------------------------------------: |
| Use more/less data points of the vector field    | <kbd>+/-</kbd>                              |
| Regular Visualisation Mode                       | <kbd>1</kbd>                                |
| Isosurface Visualisation Mode                    | <kbd>2</kbd>                                |
| Slab (X,Y,Z) Visualisation Mode                  | <kbd>3</kbd> <kbd>4</kbd> <kbd>5</kbd>      |
| Cycle Visualisation Mode                         | <kbd>/</kbd>                                |
| Move Slab                                        | <kbd>,</kbd> / <kbd>.</kbd> ( <kbd>Shift</kbd> to go faster) |

### Manipulate the chain of images

| Effect                                           | Keystroke                                   |
| ------------------------------------------------ | :-----------------------------------------: |
| Switch between images and chains                 | <kbd>&larr;</kbd> <kbd>&uarr;</kbd> <kbd>&rarr;</kbd> <kbd>&darr;</kbd> |
| Cut image                                        | <kbd>Ctrl</kbd>+<kbd>X</kbd>                |
| Copy image                                       | <kbd>Ctrl</kbd>+<kbd>C</kbd>                |
| Paste image at current index                     | <kbd>Ctrl</kbd>+<kbd>V</kbd>                |
| Insert left/right of current index               | <kbd>Ctrl</kbd>+<kbd>&larr;</kbd> / <kbd>&rarr;</kbd> |
| Delete image                                     | <kbd>Del</kbd>                              |



---

[Home](Readme.md)