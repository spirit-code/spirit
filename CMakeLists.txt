######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Build Flags #######################
### CMake Verbosity
SET( SPIRIT_PRINT_SOURCES     OFF  CACHE BOOL "Print Spirit Headers and Sources from CMake." )
### These decide which projects are built
SET( SPIRIT_BUILD_FOR_JS      OFF  CACHE BOOL "Build the JavaScript library." )
SET( SPIRIT_BUILD_FOR_JULIA   OFF  CACHE BOOL "Build the shared library for Julia." )
SET( SPIRIT_BUILD_FOR_PYTHON  ON   CACHE BOOL "Build the shared library for Python." )
SET( SPIRIT_BUILD_FOR_CXX     ON   CACHE BOOL "Build the static library for C++ applications" )
### Feature switches for Spirit
SET( SPIRIT_ENABLE_PINNING    OFF  CACHE BOOL "Enable pinning individual or rows of spins." )
### Options for Spirit
SET( SPIRIT_BUILD_TEST        ON   CACHE BOOL "Build unit tests for the Spirit library." )
SET( SPIRIT_USE_CUDA          OFF  CACHE BOOL "Use CUDA to speed up certain parts of the code." )
SET( SPIRIT_USE_THREADS       OFF  CACHE BOOL "Use std threads to speed up certain parts of the code." )
### Set the scalar type used in the Spirit library
set( SPIRIT_SCALAR_TYPE double )
#############################################
### CMake Verbosity
option( PRINT_SOURCES    "Print Headers and Sources from Cmake."                        OFF )
### Decide CXX UI
option( UI_CXX_USE_QT    "Build the QT User Interface instead of console version."      ON  )
### Bundle option
option( OSX_BUNDLE_APP   "Bundle the executable with its dependencies into a .app."     OFF )
### Option for building on the IFF cluster
option( USER_PATHS_IFF   "Use the compiler and library paths etc. for the IFF Cluster." OFF )
#############################################


#############################################
if( SPIRIT_BUILD_TEST )
    enable_testing()
endif()
#############################################
if (SPIRIT_USE_CUDA)
    MESSAGE( STATUS ">> -------------- USING CUDA --------------" )
    ### Set cuda toolkit path 
    if( (NOT CUDA_TOOLKIT_ROOT_DIR) )
		if (APPLE OR UNIX)
        	set( CUDA_TOOLKIT_ROOT_DIR /opt/cuda )
		elseif (WIN32)
			set( CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/" )
			MESSAGE( STATUS ">> We are on Windows... CUDA Path may need to be set..." )
		endif()
    endif()
    if (APPLE OR UNIX)
        set(CMAKE_C_COMPILER   ${CUDA_TOOLKIT_ROOT_DIR}/bin/gcc)
        set(CMAKE_CXX_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/g++)
    elseif (WIN32)
        # MESSAGE( STATUS ">> We are on Windows... CUDA untested" )
    endif()
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



######### Set all Flags and Options #########
### Choose Compiler
include(ChooseCompiler)
### Project Name
PROJECT(spirit)
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
# set_property(TARGET libqhullstatic_r PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
if (WIN32)
	set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_MINSIZEREL ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/MinSizeRel/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/RelWithDebInfo/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()
add_dependencies(libqhullstatic_r qhull)
### qhullcpp library
add_library(libqhullcpp STATIC IMPORTED)
set_property(TARGET libqhullcpp PROPERTY INTERFACE_LINK_LIBRARIES libqhullstatic_r)
# set_property(TARGET libqhullcpp PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
if (WIN32)
	set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_MINSIZEREL ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/MinSizeRel/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/RelWithDebInfo/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
endif ()
add_dependencies(libqhullcpp qhull)
set(qhull_LIBS libqhullcpp)
### Add corresponding include directories
set(qhull_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/thirdparty-install/include;${CMAKE_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
include_directories( ${qhull_INCLUDE_DIRS} )
#############################################


######### Add Subdirectory Projects #########
### Spirit library is built in any case
add_subdirectory( core )
### CXX UI
if( SPIRIT_BUILD_FOR_CXX )
	if ( UI_CXX_USE_QT )
		add_subdirectory( VFRendering )
		add_subdirectory( ui-qt )
	else()
		add_subdirectory( ui-console )
	endif()
endif()
### Web UI
if( SPIRIT_BUILD_FOR_JS )
	add_subdirectory( ui-web )
endif()
#############################################


######### Write VERSION.txt #################
file(WRITE "${CMAKE_SOURCE_DIR}/VERSION.txt" "${SPIRIT_META_NAME_VERSION}")
#############################################
