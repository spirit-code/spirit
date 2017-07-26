Building Spirit's Framework Components
======================================

The **Spirit** framework is designed to run across different platforms
and so the build process is set up with `CMake`, which will generate
the appropriate build scripts for each platform.

Please be aware that our CMake scripts are written for our use cases and
**you may need to adapt some paths and options in the Root CMakeLists.txt**.



&nbsp;



Contents
--------

1. [General Build Process](#Build)
2. [Core Library](#Core)
4. [Desktop User Interface](#QT)

---------------------------------------------



&nbsp;



General Build Process <a name="Build"></a>
---------------------------------------------

>The following assumes you are in the Spirit root directory.

### Options
There are some important **Options** you may need to consider.
You can find them under *### Build Flags ###* in the **Root [CMakeLists.txt](../CMakeLists.txt)**.
Otherwise, the developers' defaults will be used.

Some **Paths** you can set under *### User Paths ###* (just uncomment the corresponding line) are:

| CMake Variable                         | Use |
| :------------------------------------: | :-: |
| USER_COMPILER_C<br />USER_COMPILER_CXX | Name of the compiler you wish to use    |
| USER_PATH_COMPILER                     | Directory your compiler is located in   |
| USER_PATHS_IFF                         | use the default IFF (FZJ) cluster paths |

### Clean
Clear the build directory using

	./clean.sh
	or
	rm -rf build && mkdir build

Further helper scripts for clean-up are `clean_log.sh`, `clean_output.sh`, 
	
### Generate Build Files
`./cmake.sh` lets cmake generate makefiles for your system inside a 'build' folder.
Simply call

	./cmake.sh
	or
	cd build && cmake .. && cd ..

Passing `-debug` to the script will cause it to create a debug configuration,
meaning that you will be able to properly debug the entire application.	

On **Windows** (no MSys) you can simply use the git bash to do this or use the CMake GUI.
When using MSys etc., CMake will create corresponding MSys makefiles.

### Building the Projects
To execute the build and linking of the executable, simply call

	./make.sh -jN
	or
	cd build && make -jN && cd ..

where `-jN` is optional, with `N` the number of parallel build processes you want to create.

On **Windows** (no MSys), CMake will by default have generated a Visual Studio Solution.
Open the generated Solution in the Visual Studio IDE and build it there.

### Running the Unit Tests
We use `CMake`s `CTest` for unit testing. You can run

	ctest.sh
	or
	cd build && ctest --output-on-failure && cd ..

or execute any of the test executables manually.
To execute the tests from the Visual Studio IDE, simply rebuild the `RUN_TESTS` project.


### Installing Components

This is not yet supported! however, you can already run

	./install.sh
	or
	cd build && make install && cd ..

Which on OSX should build a .app bundle.

---------------------------------------------



&nbsp;



Core Library <a name="Core"></a>
---------------------------------------------

For detailed build instructions concerning the standalone core library
or how to include it in your own project, see [core/docs/BUILD.md](../core/docs/BUILD.md).
* Shared and static library
* Python bindings
* Julia bindings
* Transpiling to JavaScript
* Unit Tests

The **Root [CMakeLists.txt](../CMakeLists.txt)** has a few options you can set:

|  CMake Options          | Use |
| :---------------------: | :-: |
| SPIRIT_USE_CUDA         | Use CUDA to speed up numerically intensive parts of the core |
| SPIRIT_SCALAR_TYPE      | Should be e.g. `double` or `float`. Sets the C++ type for scalar variables, arrays etc. |
|  | |
| SPIRIT_BUILD_TEST       | Build unit tests for the core library |
| SPIRIT_BUILD_FOR_CXX    | Build the static library for C++ applications |
| SPIRIT_BUILD_FOR_JULIA  | Build the shared library for Julia |
| SPIRIT_BUILD_FOR_PYTHON | Build the shared library for Python |
| SPIRIT_BUILD_FOR_JS     | Build the JavaScript library (uses a different toolchain!) |

---------------------------------------------



&nbsp;



Desktop User Interface <a name="QT"></a>
---------------------------------------------

|  Dependencies  | Versions |
| :------------: | -------- |
| OpenGL Drivers | >= 3.3   |
| CMake          | >= 3.5   |
| QT             | 5.7 including QT-Charts |

**Note** that in order to build with QT as a dependency on Windows, you may need to add
`path/to/qt/qtbase/bin` to your PATH variable.

Necessary OpenGL drivers *should* be available through the regular drivers for any
remotely modern graphics card.

|  CMake Options       | Use |
| :------------------: | :-: |
| SPIRIT_BUILD_FOR_CXX | Build the C++ interfaces (console or QT) instead of others |
| UI_CXX_USE_QT        | Build qt user interface instead of console version |
| USER_PATH_QT         | The path to your CMake installation |
| BUNDLE_APP           | On OSX, create .app bundle (not yet fully functional) |



---

[Home](Readme.md)