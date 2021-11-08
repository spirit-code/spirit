Building Spirit on Windows
======================================

**Binary packages are currently not provided!**
Therefore, you need to build the Spirit core library
or the desktop user interface yourself.

The **Spirit** framework is designed to run across different
platforms and uses `CMake` for its build process, which will
generate the appropriate build scripts for each platform.

Note that you may use the CMake GUI to configure the options
or use the command line, for example through the git bash.


Core library
--------------------------------------

**Requirements**

- cmake >= 3.10
- compiler with C++14 support, e.g. msvc 19.10 (VS 2017, version 15.1)

**Build**

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
--------------------------------------

By default, the Qt desktop GUI will try to build. The corresponding
CMake option is `SPIRIT_UI_CXX_USE_QT`. To build the immediate mode
(IM GUI) instead, use `SPIRIT_UI_USE_IMGUI=ON`.

**Additional requirements**

- OpenGL drivers >= 3.3
- The Qt GUI requires Qt >= 5.7 (including qt-charts)

Necessary OpenGL drivers *should* be available through the regular drivers for any
remotely modern graphics card.

**Note** that in order to build with Qt as a dependency on Windows, you may need to add
`path/to/qt/qtbase/bin` to your PATH variable.


Python package
--------------------------------------

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
--------------------------------------

Using OpenMP on Windows is not officially supported.
While you can use other compiler/implementation combinations, the
build process tends to be nontrivial. We recommend using LLVM/clang.


CUDA backend
--------------------------------------

The CUDA backend can be used to speed up calculations by
using a GPU.

At least version 8 of the CUDA toolkit is required and the
GPU needs compute capability 3.0 or higher!

Note that **the GUI cannot be used on the CUDA backend on Windows**!
(see the CUDA programming guide:
[coherency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-coherency-hd)
and
[requirements](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements))

*Note:* the precision of the core will be automatically set
to `float` in order to avoid the performance cost of `double`
precision operations on GPUs.

**Build**

You need to set the corresponding `SPIRIT_USE_CUDA` CMake
variable, e.g. by calling

```
cd build
cmake -DSPIRIT_USE_CUDA=ON ..
cd ..
```

or by setting the option in the CMake GUI and re-generating.

You may additionally need to
- manually set the host compiler
  ("C:/Program Files (x86)/.../bin/cl.exe)
- manually set the CUDA Toolkit directory in the CMake GUI or
  pass the `CUDA_TOOLKIT_ROOT_DIR` to cmake or edit it in the
  root CMakeLists.txt
- select the appropriate arch for your GPU using the
  `SPIRIT_CUDA_ARCH` CMake variable
- add the CUDA Toolkit directory to the Windows PATH, so that
  the libraries will be found when the code is executed


Web apps
--------------------------------------

Using emscripten, the Spirit core library and ImGUI app can be built to
web assembly (wasm), meaning they can be run in the browser.

The CMake options you need to set to `ON` is called `SPIRIT_BUILD_FOR_JS`
and `SPIRIT_UI_USE_IMGUI`.

The build process on Windows has not been tested by us
and we do not officially support it.


Further build configuration options
--------------------------------------

More options than described above are available,
allowing for example to deactivate building the
Python library or the unit tests.

To list all available build options, call
```
cd build
cmake -LH ..
```
The build options of Spirit all start with `SPIRIT_`.


Installation
--------------------------------------

*Please note that the following steps are not well-tested!*

This step is not needed, unless you wish to redistribute spirit.
A system-wide installation is not supported.

Setting the CMake option `SPIRIT_BUNDLE_APP=ON` will cause the install step
to create a redistibutable folder containing all the necessary binaries.

If you then trigger the packaging step, a zip of this folder will be
generated.