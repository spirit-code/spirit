######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Build Flags #######################
### These decide which projects are built
option( BUILD_UI_WEB   "Build the JavaScript library for the Web UI."              OFF )
option( UI_USE_QT      "Build the QT User Interface instead of console version."   ON  )
### Bundle option
option( OSX_BUNDLE_APP "Bundle the executable with its dependencies into a .app."  OFF )
### CMake Verbosity
option( PRINT_SOURCES  "Print Headers and Sources from Cmake."                     OFF )
#############################################


######### Paths #############################
### Set the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
### Set thirdparty include dirs
set( GLAD_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/thirdparty/glad/include)
set( GLM_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/thirdparty )
#############################################


######### Get git revision ##################
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(SUBSTRING "${GIT_SHA1}" 0 12 GIT_REV)
#############################################


######### Meta information about the project
set(META_PROJECT_NAME         "MonoSpin")
set(META_PROJECT_DESCRIPTION  "Modular Numerical Optimizations Spin Code")
set(META_AUTHOR_ORGANIZATION  "")
set(META_AUTHOR_DOMAIN        "https://github.com/PGI-1/monospin")
set(META_AUTHOR_MAINTAINER    "g.mueller@fz-juelich.de")
set(META_VERSION_MAJOR        "1")
set(META_VERSION_MINOR        "0")
set(META_VERSION_PATCH        "0")
set(META_VERSION_REVISION     "${GIT_REV}")
set(META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}")
set(META_NAME_VERSION         "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})")
#############################################


######### Set all Flags and Options #########
### Choose Compiler
include(ChooseCompiler)
### Project Name
PROJECT(${META_PROJECT_NAME})
### Platform-specific Flags
include(Platforms)
### Compiler-specific Flags
include(CompilerFlags)
### Prevent in-source builds
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated VERSION.txt
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


######### Write VERSION.txt #################
file(WRITE "${CMAKE_SOURCE_DIR}/VERSION.txt" "${META_NAME_VERSION}")
#############################################
