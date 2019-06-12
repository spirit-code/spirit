Building Spirit on Windows
======================================

**Binary packages are currently not provided!**
Therefore, you need to build the Spirit core library
or the desktop user interface yourself.

The **Spirit** framework is designed to run across different
platforms and uses `CMake` for its build process, which will
generate the appropriate build scripts for each platform.
To list all available build options, call
```
cd build
cmake -LH ..
```


Core library
-------------------------

The Visual Studio Version needs to be specified and it usually
makes sense to specify 64bit, as it otherwise defaults to 32bit.
The version number and year may be different for you, Win64
can be appended to any of them.  Execute `cmake -G` to get
a listing of the available generators.

```
# enter the top-level Spirit directory
$ cd spirit

# make a build directory and enter that
$ mkdir build
$ cd build

# Generate a solution file
$ cmake -G "Visual Studio 14 2015 Win64" ..

# Either open the .sln with Visual Studio, or run
$ cmake --build . --config Release
```

You can also open the CMake GUI and configure and generate
the project solution there. The solution file can be opened
and built using Visual Studio, which is especially useful
for debugging.


Desktop GUI
-------------------------

By default, the desktop GUI will try to build. The corresponding
CMake option is `SPIRIT_UI_CXX_USE_QT`.

### Additional requirements

- Qt >= 5.7 (including qt-charts)
- OpenGL drivers >= 3.3

Necessary OpenGL drivers *should* be available through the regular drivers for any
remotely modern graphics card.

**Note** that in order to build with Qt as a dependency on Windows, you may need to add
`path/to/qt/qtbase/bin` to your PATH variable.


Python package
-------------------------

The Python package is built by default. The corresponding
CMake option is `SPIRIT_BUILD_FOR_PYTHON`.
The package is then located at `core/python`. You can then
- make it locatable, e.g. by adding `path/to/spirit/core/python` to your
`PYTHONPATH`
- `cd core/python` and `pip install -e . --user` to install it

Alternatively, the most recent release version can be
installed from the [official package](https://pypi.org/project/spirit/),
e.g. `pip install spirit --user`.


OpenMP backend
-------------------------

Using OpenMP on Windows is not officially supported.
While it is possible to use it, the build process is
nontrivial.


CUDA backend
-------------------------

The CUDA backend can be used to speed up calculations by
using a GPU.

At least version 8 of the CUDA toolkit is required!

*Note:* the precision of the core will be automatically set
to `float` in order to avoid the performance cost of `double`
precision operations on GPUs.

### Build

The CMake option you need to set to `ON` is called
`SPIRIT_BUILD_FOR_JS`.

You need to set the corresponding `SPIRIT_USE_CUDA` CMake
variable, e.g. by calling

```
cd build
cmake -DSPIRIT_USE_CUDA=ON ..
cd ..
```

or by setting the option in the CMake GUI and re-generating.

You may additionally need to
- manually set the host compiler ("C:/Program Files (x86)/.../bin/cl.exe)
- select the appropriate arch for your GPU using the `SPIRIT_CUDA_ARCH` CMake variable


Web assembly library
-------------------------

Using emscripten, Spirit can be built as a Web assembly
library, meaning that it can be used e.g. from within
JavaScript.

The build process on Windows has not been tested by us
and we do not officially support it.