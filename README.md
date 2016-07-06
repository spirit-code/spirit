MonoSpin
========
**Modular Numerical Optimizations Spin Code**  
Wiki Page: https://iffwiki.fz-juelich.de/index.php/MonoSpin

<!--
![nur ein Beispiel](https://commons.wikimedia.org/wiki/File:Example_de.jpg "Beispielbild")
-->

Contents
--------
1. [Introduction](#Introduction)
2. [User Interfaces](#UserInterfaces)
3. [Branches](#Branches)
4. [Code Dependencies](#Dependencies)
5. [Installation Instructions](#Installation)

&nbsp;
 

Introduction <a name="Introduction"></a>
========================================

**Platform-independent** code with optional visualization, written in C++11.
The build process is platform-independent as well, using CMake. 
This code has been developed as a flexible solution to various use-cases, including:
* **Spin Dynamics simulations** obeying the
  [Landau-Lifschitz-Gilbert equation](https://en.wikipedia.org/wiki/Landau%E2%80%93Lifshitz%E2%80%93Gilbert_equation "Titel, der beim Überfahren mit der Maus angezeigt wird")
* Direct **Energy minimisation** of a spin system
* **Minimum Energy Path calculations** for transitions between different
  spin configurations, using the GNEB method
* Energy Landscape **Saddlepoint searches** using the MMF method

More details may be found in the [Wiki](https://iffwiki.fz-juelich.de/index.php/MonoSpin "Click me...").

&nbsp;
   

User Interfaces <a name="UserInterfaces"></a>
===================================================
The code is separated into several folders, representing the 'core' physics code
and the various user interfaces you may build.
* **core**:        Core Physics code
* **ui-console**:  Console version
* **ui-qt**:       OpenGL visualisation inside QT
* **ui-web**:      WebGL visualisation inside a website
* **gl**:          Reusable OpenGL code
* **thirdparty**:  Third party libraries the code uses

Due to this modularisation, arbitrary user interfaces may be placed on top of the core physics code.
The console version may be useful to be run on clusters, whereas the GUI-versions are useful to
provide live visualisations of the simulations to the user.
Note that the current web interface is primarily meant for educational purposes. Since the core needs to be
transpiled to JavaScript (which does not support threads) and is executed user-side, it is necessarily slower
than the regular code.

&nbsp;


Branches <a name="Branches"></a>
===================================================
We aim to adhere to the "git flow" branching model: http://nvie.com/posts/a-successful-git-branching-model/

>Release branch versions are tagged `x.x`, starting at `1.0`

&nbsp;


Code Dependencies <a name="Dependencies"></a>
=============================================

The Core does not have dependencies, except for C++11.
Due to the modular CMake Project structure, when building only a specific UI,
one does not need any libraries on which other projects depend.

Core
------------
* gcc >= 4.8.1 (C++11 stdlib)
* cmake >= 2.8.12

UI-QT
--------------------
* QT >= 5.5

In order to build with QT as a dependency, you need to have `path/to/qt/qtbase/bin` in your PATH variable.

Note that building QT can be a big pain, but usually it should work if you simply use their installers.

GL
--------------------
* OpenGL Drivers >= 3.3
* GLAD (pre-built)
* (GR? -- maybe later)

Necessary OpenGL drivers *should* be available through the regular drivers for any remotely modern graphics card.
To build GLAD, use the following:

	cd lib/glad
	cmake .
	make


Web
-----------------
* emscripten

In order to build the core.js JavaScript library, you need emscripten.
Note we have not tested this process on different machines.

&nbsp;



Installation Instructions <a name="Installation"></a>
=====================================================

>The following assumes you are in the MonoSpin root directory.

Please be aware that our CMake scripts are written for our use cases and
you may need to adapt some paths etc.

In order to build a specific UI, set the corresponding switches in the
root CMakeLists.txt.

  
Clear the build directory using

	./clean.sh
	or
	rm -rf build && mkdir build
	
Generate Build Files
--------------------
`./cmake.sh` lets cmake generate makefiles for your system inside a 'build' folder.
Simply call

	./cmake.sh
	or
	cd build && cmake .. && cd ..
	
When on pure **Windows** (no MSys etc), you can simply use the git bash to do this.
When using MSys etc., CMake will create corresponding MSys makefiles.

Building the Projects
---------------------
`./make.sh` executes the build and linking of the executable. Simply call

	./make.sh
	or
	cd build && make && cd ..

If building any C++ UI, the executable `monospin` should now be in the root folder

When on pure **Windows** (no MSys etc), instead of using `make` or `./make.sh`,
you need to open the generated Solution in Visual Studio and build it there.
The execution folder should be 'build' and file paths at runtime will be
relative to this folder.