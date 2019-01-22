Spirit Desktop UI
======================

![Logo](https://imgur.com/lGZNdop.png "Spirit Logo")

The cross-platform QT desktop user interface provides a productive tool for Spin simulations,
providing powerful real-time visualisations and access to simulation parameters,
as well as other very useful features.

See the [framework build instructions](BUILD.md) for information on how to build
the user interface on your machine.

Physics features
----------------

Insert Configurations:
- White noise
- (Anti-) Skyrmions
- Domains
- Spin Spirals

You may manipulate the Hamiltonian as well as simulation parameters and your
output file configuration:

You may start and stop simulation and directly interact with a running simulation.
- LLG Simulation: Dynamics and Minimization
- GNEB: create transitions and calculate minimum energy paths

By copying and inserting spin systems and manipulating them you may create
arbitrary transitions between different spin states to use them in GNEB calculations.
Furthermore you can choose different images to be climbing or falling images during
your calculation.


Real-time visualisation
-----------------------
This feature is most powerful for 3D systems but shows great use for the analysis
of dynamical processes and understanding what is happening in your system during
a simulation instead of post-processing your data.

- Arrows, Surface (2D/3D), Isosurface
- Spins or Eff. Field
- Every n'th arrow
- Spin Sphere
- Directional & Position filters
- Colormaps

You can also create quite complicate visualisations by combining these different features
in order to visualise complex states in 3D systems:

![Visualisation of a complicated state](http://i.imgur.com/IznxguU.png "Complicated visualisation combinating isosurface, arrows and filters")

Note that a data plot is available to visualise your chain of spin systems. It can also
show interpolated energies if you run a GNEB calculation.

![GNEB Transition Plot](http://i.imgur.com/TQpOcuh.png "Minimum energy path")


Additional features
-------------------
- Drag mode: drag, copy, insert, change radius
- Screenshot
- Read configuration or chain
- Save configuration or chain


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

### Manipulate the current images

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