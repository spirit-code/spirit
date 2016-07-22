UI - QT
------------

This is a QT5 user interface implementing the **core** and the **gl** library.
It enables the user to control parameters during simulations and provides
a live visualisation of the spin systems.

### Widgets
This UI implements *MainWindow*, which is derived from *QMainWindow* as the
central element. The simulation can be chosen and started and stopped from
there. Via the file menu, one can also import and export e.e. spin configurations.
The *Spin Widget* wraps the **gl** librarys visualisation
and provides user interaction.
The remaining widgets are used to control parameters and are packed into a
*QDockWidget*, meaning they are repositionable and can be toggled. 

### ui files
The *ui* folder contains QT-specific xml files, which can be edited using
*QT Creator* or *QT Designer*. 