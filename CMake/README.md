This folder contains some CMake modules which allow to use some nice extra features.

### ChooseCompiler
This is currently quite specific to the developers' machines.
You will likely want to edit this.

### CompilerFlags
This is made to decide which flags to give to the compiler.
It is especially necessary for emcc.

### Emscripten
This is a toolchain file for emcc.

### Platforms
Currently, this simply tells you which platform you are on.

### GetGitRevisionDescription
This enables you to inquire the hash of the current commit if you are in a git repo.
If you are not in a git repo or git is not installed, you will get something like "0000000".

### Version
Version.h.in is a template file with which CMake can generate a header file which `#define`s
*VERSION*, *VERSION_REVISION* and *VERSION_FULL*

### working_directory
working_directory-vcxproj.user.in is a template file with which CMake can generate a Visual Studio
user file for a project, which sets the working directory. It is used to control from where VS
executes the binary when using the debugger.