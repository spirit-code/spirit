######### CMake Version ############################################
cmake_minimum_required( VERSION 3.10 )
### Distinguish between Clang and AppleClang
cmake_policy( SET CMP0025 NEW )
####################################################################



######### Build options ############################################
### CMake Verbosity
set( SPIRIT_PRINT_SOURCES     OFF  CACHE BOOL "Print Spirit Headers and Sources from CMake." )
### These decide which projects are built
set( SPIRIT_BUILD_FOR_JS      OFF  CACHE BOOL "Build the JavaScript library." )
set( SPIRIT_BUILD_FOR_JULIA   OFF  CACHE BOOL "Build the shared library for Julia." )
set( SPIRIT_BUILD_FOR_PYTHON  ON   CACHE BOOL "Build the shared library for Python." )
set( SPIRIT_BUILD_FOR_CXX     ON   CACHE BOOL "Build the static library for C++ applications" )
### Feature switches for Spirit
set( SPIRIT_ENABLE_PINNING    OFF  CACHE BOOL "Enable pinning individual or rows of spins." )
set( SPIRIT_ENABLE_DEFECTS    OFF  CACHE BOOL "Enable defects and disorder in the lattice." )
### Options for Spirit
set( SPIRIT_BUILD_TEST        ON   CACHE BOOL "Build unit tests for the Spirit library." )
set( SPIRIT_TEST_COVERAGE     OFF  CACHE BOOL "Build in debug mode with special flags for coverage checks." )
set( SPIRIT_USE_CUDA          OFF  CACHE BOOL "Use CUDA to speed up certain parts of the code." )
set( SPIRIT_USE_OPENMP        OFF  CACHE BOOL "Use OpenMP to speed up certain parts of the code." )
set( SPIRIT_USE_THREADS       OFF  CACHE BOOL "Use std threads to speed up certain parts of the code." )
set( SPIRIT_USE_FFTW          ON   CACHE BOOL "If available, use the FFTW library instead of kissFFT." )
### Set the scalar type used in the Spirit library
set( SPIRIT_SCALAR_TYPE "double" CACHE STRING "The scalar type to be used in the Spirit library." )
### Set the compute capability for CUDA compilation
set( SPIRIT_CUDA_ARCH   "sm_60"  CACHE STRING "The CUDA compute architecture to use in case of a CUDA build." )
####################################################################
### CMake Verbosity
option( SPIRIT_PRINT_SOURCES    "Print Headers and Sources from Cmake."                          OFF )
### Decide UI
option( SPIRIT_UI_USE_IMGUI     "Build the ImGUI user interface instead of the console version." OFF )
option( SPIRIT_UI_CXX_USE_QT    "Build the QT user interface instead of the console version."    ON  )
### Bundle option
option( SPIRIT_BUNDLE_APP       "On installation, bundle the executable with its dependencies."  OFF )
### Option for building on the IFF cluster
option( SPIRIT_USER_PATHS_IFF   "Use the compiler and library paths etc. for the IFF Cluster."   OFF )
####################################################################



####################################################################
### Set a default build type in case none is passed
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    message( STATUS ">> Setting build type to 'Release' as none was specified." )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    # Set the possible values of build type for cmake-gui
    set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
elseif( CMAKE_BUILD_TYPE )
    message( STATUS ">> Did not need to set build type, using: ${CMAKE_BUILD_TYPE}" )
else()
    message( STATUS ">> Did not need to set build type. Configuration types: ${CMAKE_CONFIGURATION_TYPES}" )
endif()
### Set a default install directory in case none is passed
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR NOT CMAKE_INSTALL_PREFIX )
    set( CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
    message( STATUS ">> No installation directory given. Using: '${CMAKE_INSTALL_PREFIX}'" )
else()
    message( STATUS ">> Installation directory given: '${CMAKE_INSTALL_PREFIX}'" )
endif()
### Prevent in-source builds
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated VERSION.txt
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )
### Position independent code
set( CMAKE_POSITION_INDEPENDENT_CODE ON )
### We need at least C++14
set( CMAKE_CXX_STANDARD 14 )
### Set the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
####################################################################



####################################################################
### Depending on compiler versions it may be necessary to specify
### the compiler. Either pass them in via command-line or use
### the CUDA_TOOLKIT_ROOT_DIR variable.
if( SPIRIT_USE_CUDA )
    ### Deactivate OpenMP
    set( SPIRIT_USE_OPENMP OFF )
    ### Set cuda toolkit path
    if( NOT CUDA_TOOLKIT_ROOT_DIR )
        if( APPLE OR UNIX )
            set( CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda )
            # set( CUDA_TOOLKIT_ROOT_DIR /opt/cuda )
        elseif( WIN32 )
            set( CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/" )
            message( WARNING ">> We are on Windows... CUDA_TOOLKIT_ROOT_DIR may need to be passed to cmake..." )
        endif()
    endif()
    ### Set compilers
    if( APPLE OR UNIX )
        if( DEFINED CUDA_TOOLKIT_ROOT_DIR )
            message( STATUS ">> CUDA toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}" )
            if( NOT DEFINED CMAKE_C_COMPILER )
                message( STATUS ">> Set C compiler accordingly: ${CMAKE_C_COMPILER}" )
            endif()
            if( NOT DEFINED CMAKE_CXX_COMPILER )
                message( STATUS ">> Set CXX compiler accordingly: ${CMAKE_CXX_COMPILER}" )
            endif()
        else()
            message( STATUS ">> No CUDA toolkit root dir specified" )
        endif()
    elseif( WIN32 )
        # MESSAGE( STATUS ">> We are on Windows... CUDA untested" )
    endif()
endif()
####################################################################



######### Determine the compiler ###################################
### IFF cluster paths
if( SPIRIT_USER_PATHS_IFF )
    message( STATUS ">> Using IFF Paths" )
    ### GCC compiler
    set( USER_COMPILER_C    "gcc" )
    set( USER_COMPILER_CXX  "g++" )
    set( USER_PATH_COMPILER "/usr/local/gcc6/bin" )
    ### Qt location
    set( USER_PATH_QT       "/usr/local/qt5" )
endif()
### User Paths
### Set the following if you do not want cmake to choose your compiler
# set( USER_COMPILER_C    "gcc" )
# set( USER_COMPILER_CXX  "g++" )
# set( USER_PATH_COMPILER "/usr/bin" )
### Set the following if you need cmake to find your Qt installation
# set( USER_PATH_QT       "~/QT/5.7" )
### Choose the right compiler
include( ChooseCompiler )
####################################################################



######### Project name #############################################
project( spirit )
### Print compiler info
message( STATUS ">> Please check the CMAKE_CXX_COMPILER to make sure it's the right one" )
message( STATUS ">> CMAKE_C_COMPILER:   ${CMAKE_C_COMPILER}" )
message( STATUS ">> CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}" )
####################################################################



######### Platform-specific Flags ##################################
### Platform-specific flags
if( APPLE )
    set( PLATFORM_NAME "Apple" )
    if( SPIRIT_BUNDLE_APP )
        message( STATUS ">> Going to create a .app bundle" )
        set( OS_BUNDLE MACOSX_BUNDLE )
    endif()
elseif( UNIX )
    set( PLATFORM_NAME "UNIX" )
elseif( WIN32 )
    set( PLATFORM_NAME "Win32" )
    add_compile_definitions( NOMINMAX _CRT_SECURE_NO_WARNINGS )
endif()
### Compiler-specific flags
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1 )
        message( FATAL_ERROR "GCC version must be at least 5.1!" )
    endif()
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" )
    ### Disable unnecessary warnings on Windows, such as C4996 and C4267, C4244
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /wd4018 /wd4244 /wd4267 /wd4661 /wd4996" )
endif()
###
message( STATUS ">> We are on the platform: ${PLATFORM_NAME}" )
message( STATUS ">> CMAKE_CXX_COMPILER_ID:  ${CMAKE_CXX_COMPILER_ID}" )
message( STATUS ">> CMAKE_CXX_FLAGS:        ${CMAKE_CXX_FLAGS}" )
message( STATUS ">> CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}" )
####################################################################



####################################################################
if( SPIRIT_BUILD_TEST )
    enable_testing()
endif()
###
if( SPIRIT_USE_CUDA )
    enable_language( CUDA )
endif()
###
if( SPIRIT_USE_OPENMP )
    include( FindOpenMP )
    if( OPENMP_FOUND )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
        set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
    endif()
endif()
###
if( SPIRIT_SKIP_HTST )
    add_definitions( "-DSPIRIT_SKIP_HTST" )
    message( STATUS ">> Skipping compilation of HTST!" )
endif()
###
if( SPIRIT_BUILD_FOR_JS )
    set( SPIRIT_BUILD_FOR_CXX OFF )
    set( SPIRIT_USE_CUDA      OFF )
    set( SPIRIT_USE_OPENMP    OFF )
    set( SPIRIT_UI_CXX_USE_QT OFF )
    set( SPIRIT_UI_USE_IMGUI  ON  )

    if( ${CMAKE_SYSTEM_NAME} MATCHES "Emscripten" )
        message( WARNING "You set SPIRIT_BUILD_FOR_JS to ON but your are not using emscripten. That might not work" )
    endif()
endif()
### Need to use pthread if emscripten is used
if( SPIRIT_BUILD_FOR_JS AND SPIRIT_UI_USE_IMGUI )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -s USE_PTHREADS=1 -s USE_PTHREADS=1 -s WASM=1" )
endif()
####################################################################



######### Add subdirectory projects ################################
add_subdirectory( thirdparty/qhull )
set( qhull_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/qhull/src
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/qhull/src/libqhullcpp )
if( CMAKE_BUILD_TYPE MATCHES "[dD]ebug" )
    set( qhull_LIBS qhullcpp_d qhullstatic_rd)
else()
    set( qhull_LIBS qhullcpp qhullstatic_r )
endif()
### Spirit library is built in any case
add_subdirectory( core )
### Web UI
if( SPIRIT_BUILD_FOR_JS )
    if( SPIRIT_UI_USE_IMGUI )
        add_definitions( -DSPIRIT_UI_USE_IMGUI )
    endif()
    add_subdirectory( VFRendering )
    add_subdirectory( ui-cpp )
    add_subdirectory( ui-web )
### CXX UI
elseif( SPIRIT_BUILD_FOR_CXX )
    if( SPIRIT_UI_CXX_USE_QT )
        add_definitions( -DSPIRIT_UI_CXX_USE_QT )
    elseif( SPIRIT_UI_USE_IMGUI )
        add_definitions( -DSPIRIT_UI_USE_IMGUI )
    endif()
    add_subdirectory( VFRendering )
    add_subdirectory( ui-cpp )
endif()
####################################################################



################ Install ###########################################
install( DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/docs/
         DESTINATION docs/Spirit/
         COMPONENT spirit_root_files )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/README.md
               ${CMAKE_CURRENT_LIST_DIR}/VERSION.txt
         DESTINATION ./
         COMPONENT spirit_root_files )

if( SPIRIT_BUILD_FOR_CXX )
    install( DIRECTORY input DESTINATION bin COMPONENT spirit_root_files )
endif()

install( FILES ${CMAKE_CURRENT_LIST_DIR}/LICENSE.txt DESTINATION ./ COMPONENT spirit_licenses )

install( FILES ${CMAKE_CURRENT_LIST_DIR}/thirdparty/qhull/COPYING.txt DESTINATION licenses RENAME qhull.txt COMPONENT spirit_licenses )

install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/cub/LICENSE.TXT   DESTINATION licenses RENAME cub.txt       COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/Eigen/COPYING.BSD DESTINATION licenses RENAME eigen.txt     COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/fmt/LICENSE.rst   DESTINATION licenses RENAME fmt.rst       COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/kiss_fft/COPYING  DESTINATION licenses RENAME kiss_fft.txt  COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/ovf/README.md     DESTINATION licenses RENAME ovf.md        COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/spectra/LICENSE   DESTINATION licenses RENAME spectra.txt   COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/core/thirdparty/termcolor/LICENSE DESTINATION licenses RENAME termcolor.txt COMPONENT spirit_licenses )

install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/thirdparty/Lyra/LICENSE.txt DESTINATION licenses RENAME lyra.txt COMPONENT spirit_licenses )

install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/filesystem/LICENSE       DESTINATION licenses RENAME filesystem.txt       COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/glad/LICENSE             DESTINATION licenses RENAME glad.md              COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/glfw/LICENSE.md          DESTINATION licenses RENAME glfw.md              COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/imgui/LICENSE.txt        DESTINATION licenses RENAME imgui.txt            COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/implot/LICENSE           DESTINATION licenses RENAME implot.txt           COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/json/LICENSE.MIT         DESTINATION licenses RENAME json.txt             COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/nativefiledialog/LICENSE DESTINATION licenses RENAME nativefiledialog.txt COMPONENT spirit_licenses )
install( FILES ${CMAKE_CURRENT_LIST_DIR}/ui-cpp/ui-imgui/thirdparty/stb/LICENSE              DESTINATION licenses RENAME stb.txt              COMPONENT spirit_licenses )
####################################################################



######### Write VERSION.txt ########################################
file( WRITE "${CMAKE_SOURCE_DIR}/VERSION.txt" "${SPIRIT_META_NAME_VERSION}" )
####################################################################