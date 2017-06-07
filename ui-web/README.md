UI - Web
------------

This interface is implemented as a website, incorporating the **core** library
as a JavaScipt library. To this end, the library needs to be transpiled with
Emscripten.

### Transpiling the Core library
The root *CMakeLists.txt* includes an option called `SPIRIT_BUILD_FOR_JS`, which you
can switch on. This should load the necessary toolchain file, the compiler
flags and linker flags.
This has been tested only on the developer's machines (excluding Windows)
and may well initially fail elsewhere.