UI - QT
------------

This is a QT5 user interface making use of the **core** and the **VFRendering**
libraries. It enables the user to control parameters during simulations and provides
a powerful live visualisation of the spin systems.

### Widgets
The available calculation methods and optimizers, starting and stopping the calculation
as well as the switching between images are implemented in `ControlWidget`. Via the file
menu, one can import and export e.g. spin configurations.
The `Spin Widget` wraps the **VFRendering** library's visualisation capabilities
and provides user interaction.
The remaining widgets are used to control parameters and view output. They are packed
into a `QDockWidget`, meaning they are repositionable and can be toggled. 

### ui files
The *ui* folder contains QT-specific xml files, which can be edited using
*QT Creator* or *QT Designer*. 