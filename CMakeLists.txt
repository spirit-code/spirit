######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Build Flags #######################
### These decide which projects are built
option( BUILD_UI_WEB     "Build the JavaScript library for the Web UI."                 OFF )
option( BUILD_UI_JULIA   "Build the shared library for the Julia UI."                   OFF )
option( BUILD_UI_PYTHON  "Build the shared library for the Python UI."                  ON  )
option( BUILD_UI_CXX     "Build the C++ UIs (console or QT)"                            ON  )
option( UI_CXX_USE_QT    "Build the QT User Interface instead of console version."      ON  )
### Bundle option
option( OSX_BUNDLE_APP   "Bundle the executable with its dependencies into a .app."     OFF )
### CMake Verbosity
option( PRINT_SOURCES    "Print Headers and Sources from Cmake."                        OFF )
### Option for building on the IFF cluster
option( USER_PATHS_IFF   "Use the compiler and library paths etc. for the IFF Cluster." OFF )
### Options for the usage of CUDA and threads
option( CORE_USE_CUDA    "Use CUDA to speed up certain parts of the code."              OFF )
option( CORE_USE_THREADS "Use std threads to speed up certain parts of the code."       OFF )
### Set the scalar type used in the core library
set( CORE_SCALAR_TYPE double )
#############################################
if( BUILD_UI_WEB )
	### UI-Web needs float
	set( CORE_SCALAR_TYPE 	float )
	### UI-Web needs to be built alone, as it
	### uses a different toolchain
	set( BUILD_UI_JULIA 	OFF )
	set( BUILD_UI_PYTHON 	OFF )
	set( BUILD_UI_CXX 		OFF )
	### Emscripten cannot use cuda or threads
	set( CORE_USE_CUDA 		OFF )
	set( CORE_USE_THREADS 	OFF )
endif()
#############################################

######### General Paths #####################
### Set the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
######### User Paths ########################
include(PathsIFF)
### Set these if you do not want cmake to choose your compiler
# set( USER_COMPILER_C    "gcc" )
# set( USER_COMPILER_CXX  "g++" )
# set( USER_PATH_COMPILER "/usr/bin" )
### Set this if you need cmake to find your QT installation
# set( USER_PATH_QT       "~/QT/5.7" )
#############################################


######### Get git revision ##################
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(SUBSTRING "${GIT_SHA1}" 0 12 GIT_REV)
#############################################


######### Meta information about the project
set(META_PROJECT_NAME         "Spirit")
set(META_PROJECT_DESCRIPTION  "Modular Spin Code")
set(META_AUTHOR_ORGANIZATION  "")
set(META_AUTHOR_DOMAIN        "https://github.com/spirit-code/spirit")
set(META_AUTHOR_MAINTAINER    "g.mueller@fz-juelich.de")
set(META_VERSION_MAJOR        "1")
set(META_VERSION_MINOR        "2")
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
### Print compiler info
MESSAGE( STATUS ">> Please check the CMAKE_CXX_COMPILER to make sure it's the right one" )
MESSAGE( STATUS ">> CMAKE_C_COMPILER:               " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:             " ${CMAKE_CXX_COMPILER} )
### Platform-specific Flags
include(Platforms)
### Compiler-specific Flags
include(CompilerFlags)
### Prevent in-source builds
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated VERSION.txt
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
### Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
### Installation Prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()
#############################################



######### External Project: QHull ###########
include(ExternalProject)
ExternalProject_add(qhull
	SOURCE_DIR    "${CMAKE_SOURCE_DIR}/thirdparty/qhull"
	# GIT_REPOSITORY https://github.com/qhull/qhull.git
	BINARY_DIR ${CMAKE_BINARY_DIR}/thirdparty-build/qhull
	# INSTALL_DIR ${CMAKE_BINARY_DIR}/thirdparty-install
	CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty-install"
	CMAKE_CACHE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true"
)
### qhullstatic reentrant library
add_library(libqhullstatic_r STATIC IMPORTED)
set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
add_dependencies(libqhullstatic_r qhull)
### qhullcpp library
add_library(libqhullcpp STATIC IMPORTED)
set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
set_property(TARGET libqhullcpp PROPERTY INTERFACE_LINK_LIBRARIES libqhullstatic_r)
add_dependencies(libqhullcpp qhull)
set(qhull_LIBS libqhullcpp)
### Add corresponding include directories
set(qhull_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/thirdparty-install/include;${CMAKE_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
include_directories( ${qhull_INCLUDE_DIRS} )
#############################################


######### Add Subdirectory Projects #########
### Core is built in any case
add_subdirectory( core )
### CXX UI
if( BUILD_UI_CXX )
	if ( UI_CXX_USE_QT )
		add_subdirectory( VFRendering )
		add_subdirectory( ui-qt )
	else()
		add_subdirectory( ui-console )
	endif()
endif()
### Web UI
if( BUILD_UI_WEB )
	add_subdirectory( ui-web )
endif()
#############################################


######### Write VERSION.txt #################
file(WRITE "${CMAKE_SOURCE_DIR}/VERSION.txt" "${META_NAME_VERSION}")
#############################################
