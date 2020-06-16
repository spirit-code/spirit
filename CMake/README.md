This folder contains some CMake modules which allow to use some nice extra features.

### ChooseCompiler
This is currently quite specific to the developers' machines.
You will likely want to edit this.

### Emscripten
This is a toolchain file for emcc.

### working_directory
working_directory-vcxproj.user.in is a template file with which CMake can generate a Visual Studio
user file for a project, which sets the working directory. It is used to control from where VS
executes the binary when using the debugger.