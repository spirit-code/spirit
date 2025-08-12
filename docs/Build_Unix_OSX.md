Building Spirit on Unix/OSX
======================================

**Binary packages are currently not provided!**
Therefore, you need to build the Spirit core library
or the desktop user interface yourself.

The **Spirit** framework is designed to run across different
platforms and uses `CMake` for its build process, which will
generate the appropriate build scripts for each platform.

Core library
--------------------------------------

**Requirements**

- cmake >= 3.18
- compiler with C++17 support, e.g. gcc >= 13.2

**Build**

CMake is used to automatically generate makefiles.

```sh
# enter the top-level Spirit directory
$ cd spirit

# Generate makefiles
$ cmake -S . -B build

# Build
$ cmake --build build
```

For running the build in parallel you can pass the `-j`
option at the end of the build command.

To manually specify the build type (default is 'Release'),
call `cmake --build build --config Release` to build instead.


Desktop GUI
--------------------------------------

By default, the Qt desktop GUI will try to build. The corresponding
CMake option is `SPIRIT_UI_CXX_USE_QT`. To build the immediate mode
(IM GUI) instead, use `SPIRIT_UI_USE_IMGUI=ON`.

### Additional requirements

- OpenGL drivers >= 3.3
- On Linux, the IM GUI requires `xorg-dev` or `wayland-dev`, `libglu1-mesa-dev` or equivalent and `libgtk-3-dev`
- The Qt GUI requires Qt >= 5.7 (including qt-charts)

Necessary OpenGL drivers *should* be available through the regular drivers
for any remotely modern graphics card.


Python package
--------------------------------------

The Python package is built by default. The corresponding
CMake option is `SPIRIT_BUILD_FOR_PYTHON`.
The package is then located at `core/python`. You can then
- make it locatable, e.g. by adding `path/to/spirit/core/python` to your
`PYTHONPATH`
- `pip install -e 'core/python' --user` to install it

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

```sh
cmake -S . -B build -DSPIRIT_USE_OPENMP=ON
```


STDPAR backend
--------------------------------------

C++17 offers parallelization support for STL algorithms that can be utilized in Spirit.
This requires that the compiler and STL implementation support this feature.
On most platforms this works out of the box when using `gcc` as the compiler.
In rare cases you might have to install a 'Threading Building Blocks' library like `oneTBB`.

**Build**

You need to set the `SPIRIT_USE_STDPAR` CMake variable,
e.g. by calling

```sh
cmake -S . -B build -DSPIRIT_USE_STDPAR=ON
```


CUDA backend
--------------------------------------

The CUDA backend can be used to speed up calculations by
using a GPU.

Spirit uses [unified memory](https://devblogs.nvidia.com/unified-memory-cuda-beginners).
At least version 12 of the CUDA toolkit is required and the
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

```sh
cmake -S . -B build -DSPIRIT_USE_CUDA=ON ..
```

You may additionally need to
- pass the `CUDA_TOOLKIT_ROOT_DIR` to cmake or edit it in
  the root CMakeLists.txt
- select the appropriate arch for your GPU using the
  `SPIRIT_CUDA_ARCH` CMake variable


Web apps
--------------------------------------

Using emscripten, the Spirit core library and ImGUI app can be built to
web assembly (wasm), meaning they can be run in the browser.

The CMake options you need to set to `ON` is called `SPIRIT_BUILD_FOR_JS`
and `SPIRIT_UI_USE_IMGUI`.

You need to have emscripten available, meaning you might
need to source, e.g. `source /usr/local/bin/emsdkvars.sh`.

Then to build, call

```sh
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/usr/local/emsdk/emscripten/1.38.29/cmake/Modules/Platform/Emscripten.cmake
cmake --build build
```

You will then have the mobile-capable app in the ui-web folder and the
desktop app in ui-cpp/ui-imgui/webapp.


Further build configuration options
--------------------------------------

More options than described above are available,
allowing for example to deactivate building the
Python library or the unit tests.

To list all available build options, call
```
cmake -B build -S . -LH
```
The build options of Spirit all start with `SPIRIT_`.


Installation
--------------------------------------

*Please note that the following steps are not well-tested!*

This step is not needed, unless you wish to have spirit in
your system directories or to create a `.app` bundle on OSX.
You can set the installation directory during the configuration
stage, i.e.

```sh
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..
```

or point it to a local folder, e.g. `-DCMAKE_INSTALL_PREFIX:PATH=./install`.

**OSX .app bundle and installer**

If you want to create a redistributable bundle on OSX, use

```sh
cmake -S . -B build -DSPIRIT_BUNDLE_APP=ON
cmake --build build
cmake --install build
```

This will gather dependencies, such as Qt dlls, in a `.app` folder and
fix the link paths to make it redistributable. This app can be redistributed
or "installed" by placing it in your "Applications" directory.

You may need to update permissions,

```sh
chmod -R +x build/Spirit.app
```

Note that the bundle is already built with the regular `make` command.
To make it redistributable, it is necessary to use `make install`.

You can also create an installer as follows:

```sh
cmake -S . -B build -DSPIRIT_BUNDLE_APP=ON
cmake --build build -- package
```

Note that one can choose the generator as `cpack -G DragNDrop`.
