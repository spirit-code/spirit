######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Build Flags #######################
### These decide which projects are built
option( BUILD_UI_WEB  "Build the JavaScript library for the Web UI."             OFF )
option( UI_USE_QT     "Build the QT User Interface instead of console version."  ON  )
#############################################


######### Paths #############################
### Set the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
### Set thirdparty include dirs #######
set( GLAD_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/thirdparty/glad/include)
set( GLM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/thirdparty )
#############################################


######### Set all Flags and Options #########
### Choose Compiler
include(ChooseCompiler)
### Project Name
PROJECT(MonoSpin)
### Platform-specific Flags
include(Platforms)
### Compiler-specific Flags
include(CompilerFlags)
### Prevent in-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Add Subdirectory Projects #########
add_subdirectory(core)
if(NOT BUILD_UI_WEB)
	if (UI_USE_QT)
		add_subdirectory(gl)
		add_subdirectory(ui-qt)
	else()
		add_subdirectory(ui-console)
	endif()
endif()
if(BUILD_UI_WEB)
	add_subdirectory(ui-web)
endif()
#############################################