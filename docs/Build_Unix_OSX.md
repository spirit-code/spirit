Building Spirit on Unix/OSX
======================================

**Binary packages are currently not provided!**
Therefore, you need to build the Spirit core library
or the desktop user interface yourself.

The **Spirit** framework is designed to run across different
platforms and uses `CMake` for its build process, which will
generate the appropriate build scripts for each platform.

System-wide installation is not actively supported.
While you can call `make install` after building,
you may not achieve the desired results.


Core library
--------------------------------------

**Requirements**

- cmake >= 3.2
- compiler with C++11 support, e.g. gcc >= 5.1

**Build**

CMake is used to automatically generate makefiles.

```
# enter the top-level Spirit directory
$ cd spirit

# make a build directory and enter that
$ mkdir build
$ cd build

# Generate makefiles
$ cmake ..

# Build
$ make
```

Note that you can use the `-j` option of make to run the
build in parallel.

To manually specify the build type (default is 'Release'),
call `cmake --build . --config Release` instead of `make`.


Desktop GUI
--------------------------------------

By default, the desktop GUI will try to build. The corresponding
CMake option is `SPIRIT_UI_CXX_USE_QT`.

### Additional requirements

- Qt >= 5.7 (including qt-charts)
- OpenGL drivers >= 3.3

Necessary OpenGL drivers *should* be available through the regular drivers for any
remotely modern graphics card.


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

The OpenMP backend can be used to speed up calculations by
using a multicore CPU.

At least version 4.5 of OpenMP needs to be supported by your
compiler.

**Build**

You need to set the corresponding CMake variable, e.g.
by calling

```
cd build
cmake -DSPIRIT_USE_OPENMP=ON ..
cd ..
```


CUDA backend
--------------------------------------

The CUDA backend can be used to speed up calculations by
using a GPU.

Spirit uses [unified memory](https://devblogs.nvidia.com/unified-memory-cuda-beginners).
At least version 8 of the CUDA toolkit is required and the
GPU needs compute capability 3.0 or higher!

If the GUI is used, compute capability 6.0 or higher is
required! (see the CUDA programming guide:
[coherency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-coherency-hd))

Note that **the GUI cannot be used on the CUDA backend on OSX**!
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

You may additionally need to
- pass the `CUDA_TOOLKIT_ROOT_DIR` to cmake or edit it in
  the root CMakeLists.txt
- select the appropriate arch for your GPU using the
  `SPIRIT_CUDA_ARCH` CMake variable


Web assembly library
--------------------------------------

Using emscripten, Spirit can be built as a Web assembly
library, meaning that it can be used e.g. from within
JavaScript.

**Build**

The CMake option you need to set to `ON` is called
`SPIRIT_BUILD_FOR_JS`.

You need to have emscripten available, meaning you might
need to source, e.g. `source /usr/local/bin/emsdkvars.sh`.

Then to build, call

```
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/usr/local/emsdk/emscripten/1.38.29/cmake/Modules/Platform/Emscripten.cmake
make
cd ..
```


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