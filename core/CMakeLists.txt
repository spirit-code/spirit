MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Spirit --------------------------------------- <<" )


######### CMake Version ############################################
cmake_minimum_required( VERSION 3.10 FATAL_ERROR )
####################################################################


######### Build options ############################################
### CMake Verbosity
option( SPIRIT_PRINT_SOURCES    "Print Headers and Sources from Cmake."                  OFF )
### These decide which projects are built
option( SPIRIT_BUILD_FOR_JS     "Build the JavaScript library."                          OFF )
option( SPIRIT_BUILD_FOR_JULIA  "Build the shared library for Julia."                    OFF )
option( SPIRIT_BUILD_FOR_PYTHON "Build the shared library for Python."                   ON  )
option( SPIRIT_BUILD_FOR_CXX    "Build the static library for C++ applications"          ON  )
### Feature switches for Spirit
option( SPIRIT_ENABLE_PINNING   "Enable pinning individual or rows of spins."            OFF )
option( SPIRIT_ENABLE_DEFECTS   "Enable defects and disorder in the lattice."            OFF )
### Options for Spirit
option( SPIRIT_BUILD_TEST       "Build unit tests for the Spirit library."               ON  )
option( SPIRIT_TEST_COVERAGE    "Build in debug with special flags for coverage checks." OFF )
option( SPIRIT_USE_CUDA         "Use CUDA to speed up certain parts of the code."        OFF )
option( SPIRIT_USE_OPENMP       "Use OpenMP to speed up certain parts of the code."      OFF )
option( SPIRIT_USE_THREADS      "Use std threads to speed up certain parts of the code." OFF )
option( SPIRIT_USE_FFTW         "If available, use the FFTW library instead of kissFFT." ON  )
### Set the scalar type used in the Spirit library
option( SPIRIT_SCALAR_TYPE      "Use std threads to speed up certain parts of the code." "double" )
### Set the compute capability for CUDA compilation
option( SPIRIT_CUDA_ARCH        "Use std threads to speed up certain parts of the code." "sm_60"  )
####################################################################


####################################################################
#----- Add the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
#----- Basic options
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated Spirit_Defines.h
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#----- Set a default build type in case none is passed
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    message( STATUS ">> Spirit core: Setting build type to 'Release' as none was specified." )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    ### Set the possible values of build type for cmake-gui
    set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
elseif( CMAKE_BUILD_TYPE )
    message( STATUS ">> Spirit core: Did not need to set build type, using: ${CMAKE_BUILD_TYPE}" )
else()
    message( STATUS ">> Spirit core: Did not need to set build type. Configuration types: ${CMAKE_CONFIGURATION_TYPES}" )
endif()
####################################################################


######### Some restrictions on the options #########################
if( SPIRIT_USE_CUDA )
    ### If CUDA is used, we use float and and we cannot use OpenMP
    ### and we cannot build for JS or Julia
    set( SPIRIT_USE_OPENMP       OFF )
    set( SPIRIT_SCALAR_TYPE      float )
    set( SPIRIT_BUILD_FOR_JS     OFF )
    set( SPIRIT_BUILD_FOR_JULIA  OFF )
    set( SPIRIT_USE_FFTW         OFF )
    ### Enable CUDA language support
    enable_language( CUDA )
endif()
#-----
if( SPIRIT_USE_OPENMP )
    set( SPIRIT_USE_CUDA         OFF )
    set( SPIRIT_BUILD_FOR_JS     OFF )
    set( SPIRIT_BUILD_FOR_JULIA  OFF )
endif()
#-----
if( SPIRIT_BUILD_FOR_JS )
    ### UI-Web needs float
    set( SPIRIT_SCALAR_TYPE      float )
    ### UI-Web needs to be built alone, as it uses a different toolchain
    set( SPIRIT_BUILD_TEST       OFF )
    set( SPIRIT_BUILD_FOR_JULIA  OFF )
    set( SPIRIT_BUILD_FOR_PYTHON OFF )
    set( SPIRIT_BUILD_FOR_CXX    OFF )
    ### Emscripten cannot use cuda or threads
    set( SPIRIT_USE_CUDA         OFF )
    set( SPIRIT_USE_THREADS      OFF )
    set( SPIRIT_USE_FFTW         OFF )
endif()
#-----
if( SPIRIT_BUILD_TEST )
    enable_testing()
endif()
####################################################################


######### Meta information #########################################
#----- Get git revision
include( GetGitRevisionDescription )
get_git_head_revision( GIT_REFSPEC GIT_SHA1 )
string( SUBSTRING "${GIT_SHA1}" 0 13 GIT_REV )
#----- Meta information about the project
set( META_PROJECT_NAME         "Spirit" )
set( META_PROJECT_DESCRIPTION  "Optimizations and Dynamics Framework for atomistic Spin systems" )
set( META_AUTHOR_ORGANIZATION  "" )
set( META_AUTHOR_DOMAIN        "https://spirit-code.github.io" )
set( META_AUTHOR_MAINTAINER    "Gideon Mueller" )
set( META_AUTHOR_EMAIL         "g.mueller@fz-juelich.de" )
set( META_VERSION_MAJOR        "2" )
set( META_VERSION_MINOR        "1" )
set( META_VERSION_PATCH        "1" )
set( META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}" )
set( META_VERSION_REVISION     "${GIT_REV}" )
#----- Compiler
set( META_COMPILER             "${CMAKE_CXX_COMPILER_ID}" )
set( META_COMPILER_VERSION     "${CMAKE_CXX_COMPILER_VERSION}" )
set( META_COMPILER_FULL        "${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER_VERSION})" )
#----- Propagate version and name_version upwards
set( SPIRIT_META_VERSION_MAJOR ${META_VERSION_MAJOR} PARENT_SCOPE )
set( SPIRIT_META_VERSION_MINOR ${META_VERSION_MINOR} PARENT_SCOPE )
set( SPIRIT_META_VERSION_PATCH ${META_VERSION_PATCH} PARENT_SCOPE )
set( SPIRIT_META_VERSION       "${META_VERSION}" PARENT_SCOPE )
set( SPIRIT_META_NAME_VERSION  "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})" PARENT_SCOPE )
####################################################################


######### Project name #############################################
message( STATUS ">> Building ${META_PROJECT_NAME} core library v${META_VERSION} (revision ${META_VERSION_REVISION})" )
project( ${META_PROJECT_NAME} VERSION ${META_VERSION} )
####################################################################


######### Basic compiler-specific flags ############################
include(CompilerFlags)
####################################################################


######### CUDA decisions ###########################################
if( SPIRIT_USE_CUDA )
    # set( CUDA_TOOLKIT_ROOT_DIR /opt/cuda )
    find_package( CUDA 8 REQUIRED )

    # set( CUDA_PROPAGATE_HOST_FLAGS ON )
    # --std=c++14 flag may be necessary, but it is propagated from global flags...
    # if it appears twice, for some reason the compilation breaks

    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -arch=${SPIRIT_CUDA_ARCH} --expt-relaxed-constexpr --expt-extended-lambda" )
    set( SPIRIT_CUDA_LIBS "${CUDA_LIBRARIES};${CUDA_CUFFT_LIBRARIES};${CUDA_curand_LIBRARY}" )
    set( FFT_LIB          ${CUDA_CUFFT_LIBRARIES} )

    ### Deactivate CUDA warning inside Eigen such as "warning: __host__ annotation is ignored on a function("Quaternion") that is explicitly defaulted on its first declaration"
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=\"--diag_suppress=esa_on_defaulted_function_ignored\"")
    ### Display warning number when writing a warning
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--display_error_number\"" )

    set( META_COMPILER         "${META_COMPILER} and nvcc" )
    set( META_COMPILER_VERSION "${META_COMPILER_VERSION} and ${CUDA_VERSION}" )
    set( META_COMPILER_FULL    "${META_COMPILER_FULL} and nvcc (${CUDA_VERSION}) for cuda arch \\\"${SPIRIT_CUDA_ARCH}\\\"" )

    if( NOT DEFINED CMAKE_CUDA_STANDARD )
        set( CMAKE_CUDA_STANDARD 14 )
        set( CMAKE_CUDA_STANDARD_REQUIRED ON )
    endif()

    message( STATUS ">> Using CUDA. Flags: ${CUDA_NVCC_FLAGS}" )
    message( STATUS ">> CUDA toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}" )
    message( STATUS ">> CUDA libraries: ${SPIRIT_CUDA_LIBS}" )
endif()
message( STATUS ">> Compiler information: ${META_COMPILER_FULL}" )
####################################################################


######### Generate Spirit_Defines.h ################################
string( TOUPPER ${SPIRIT_SCALAR_TYPE} SPIRIT_SCALAR_TYPE_UPPERCASE )
set( THREAD_LIBS )
if( SPIRIT_USE_THREADS )
    set( THREADS_PREFER_PTHREAD_FLAG ON )
    find_package( Threads REQUIRED )
    set( THREAD_LIBS Threads::Threads )
endif()
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMake/Spirit_Defines.h.in   ${CMAKE_CURRENT_LIST_DIR}/include/Spirit_Defines.h )
####################################################################


######### OpenMP decisions #########################################
if( SPIRIT_USE_OPENMP )
    include( FindOpenMP )
    if( OPENMP_FOUND )
        set( CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}          ${OpenMP_C_FLAGS}" )
        set( CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}        ${OpenMP_CXX_FLAGS}" )
        set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
        set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_OPENMP )
        message( STATUS ">> OpenMP found." )
        message( STATUS ">> OpenMP C Flags:          ${OpenMP_C_FLAGS}" )
        message( STATUS ">> OpenMP CXX Flags:        ${OpenMP_CXX_FLAGS}" )
        message( STATUS ">> OpenMP EXE Linker Flags: ${OpenMP_EXE_LINKER_FLAGS}" )
    else( OPENMP_FOUND )
        message( WARNING ">> OpenMP could not be found." )
        set(SPIRIT_USE_OPENMP OFF)
    endif( OPENMP_FOUND )
endif()
####################################################################


######## FFT decisions #############################################

## Comment these lines in (and edit the paths) if for some reason your fftw installation is not found
#  You only need to link against libfftw3_omp if you want to use OpenMP
#  You only need to link against libfftw3f if you want to use float
# if(NOT SPIRIT_USE_CUDA)
#     set(FFT_INCLUDE_DIRS "/usr/local/include/")
#     set(FFT_LIB          "/usr/local/lib/libfftw3.so"
#                          "/usr/local/lib/libfftwf3.so"
#                          "/usr/local/lib/libfftw3_omp.so")
#     set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_FFTW )
# endif()

if( NOT SPIRIT_USE_CUDA AND NOT FFT_LIB )
    if( SPIRIT_USE_FFTW )
        include( FindFFTW )
        # Double-check for presence of needed libs
        if( FFTW_FOUND )

            set( FFT_LIB ${FFTW_LIB} )

            if( ${SPIRIT_SCALAR_TYPE} STREQUAL "float" )
                if( FFTWF_LIB )
                    set( FFT_LIB ${FFTWF_LIB} )
                else()
                    message( WARNING "Could find FFTW but not the single precision library 'libfftw3f' -> Using kissFFT fallback" )
                    set( FFTW_FOUND OFF )
                endif()
            endif()

            if( SPIRIT_USE_OPENMP )
                if( FFTW_OMP_LIB )
                    set( FFT_LIB ${FFT_LIB} ${FFTW_OMP_LIB} )
                else()
                    message( WARNING "Could find FFTW but not libfftw3_omp! -> Using kissFFT fallback" )
                    set( FFTW_FOUND OFF )
                endif()
            endif()

        else()
            message( WARNING "Could not find any FFTW libs -> Using kissFFT fallback" )
        endif()

        # Finally we know if we can use FFTW
        if( FFTW_FOUND )
            message( STATUS ">> Using FFTW" )
            set( FFT_INCLUDE_DIRS ${FFTW_INCLUDES} )
            set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_FFTW )
        else()
            set(SPIRIT_USE_FFTW OFF)
        endif()
    endif()

    if( (NOT SPIRIT_USE_FFTW) OR (NOT FFTW_FOUND) )
        message( STATUS ">> Using kissFFT" )
        if( SPIRIT_USE_OPENMP )
            message( WARNING "Using kissFFT together with OpenMP. If you intend to perform calculations including dipole-dipole interactions with OpenMP it is *strongly* recommended to use FFTW instead!" )
        endif()
        add_definitions( -Dkiss_fft_scalar=${SPIRIT_SCALAR_TYPE} )
        set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_KISSFFT )
        add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdparty/kiss_fft )
        set( FFT_LIB kiss_fft )
        set( FFT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/kiss_fft
                              ${PROJECT_SOURCE_DIR}/thirdparty/kiss_fft/tools )
    endif()
endif()

message( STATUS ">> Using FFT lib:     ${FFT_LIB}" )
message( STATUS ">> With include dirs: ${FFT_INCLUDE_DIRS}" )
####################################################################

######### Generate Version.hpp #####################################
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMake/Spirit_Version.hpp.in ${CMAKE_CURRENT_LIST_DIR}/include/utility/Version.hpp )
####################################################################

######### Dependencies that need to be built #######################
include( GetQHull )
add_subdirectory( ${PROJECT_SOURCE_DIR}/thirdparty/ovf )
####################################################################


######### Gather headers and sources ###############################
### Header Gropus
set( HEADER_SPIRIT_ROOT )
set( HEADER_SPIRIT_API )
set( HEADER_SPIRIT_DATA )
set( HEADER_SPIRIT_ENGINE )
set( HEADER_SPIRIT_IO )
set( HEADER_SPIRIT_UTILITY )
### Source Groups
set( SOURCE_SPIRIT_ROOT )
set( SOURCE_SPIRIT_API )
set( SOURCE_SPIRIT_DATA )
set( SOURCE_SPIRIT_ENGINE )
set( SOURCE_SPIRIT_IO )
set( SOURCE_SPIRIT_UTILITY )
### Add Subdirectories
add_subdirectory( ${CMAKE_CURRENT_LIST_DIR}/src )
add_subdirectory( ${CMAKE_CURRENT_LIST_DIR}/include )
####################################################################


############ Create object library #################################
add_library( ${META_PROJECT_NAME} OBJECT
    ${HEADER_SPIRIT_ROOT}
    ${HEADER_SPIRIT_DATA}
    ${HEADER_SPIRIT_ENGINE}
    ${HEADER_SPIRIT}
    ${HEADER_SPIRIT_IO}
    ${HEADER_SPIRIT_UTILITY}
    ${SOURCE_SPIRIT_ROOT}
    ${SOURCE_SPIRIT_DATA}
    ${SOURCE_SPIRIT_ENGINE}
    ${SOURCE_SPIRIT}
    ${SOURCE_SPIRIT_IO}
    ${SOURCE_SPIRIT_UTILITY} )

set_target_properties( ${META_PROJECT_NAME} PROPERTIES
    CXX_STANDARD            14
    CXX_STANDARD_REQUIRED   ON
    CXX_EXTENSIONS          OFF )

target_include_directories( ${META_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include )

target_include_directories( ${META_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/ovf/include
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/spectra/include
    ${qhull_INCLUDE_DIRS}
    ${FFT_INCLUDE_DIRS} )

target_link_libraries( ${META_PROJECT_NAME} INTERFACE
    ${qhull_LIBS} ${OVF_LIBRARIES_STATIC} ${FFT_LIB} ${THREAD_LIBS} )

### Flag needed to use fmt as header-only
set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DFMT_HEADER_ONLY )

### Need to wait for QHull to finish before building
add_dependencies( ${META_PROJECT_NAME} ${qhull_LIBS} )

### Extra CUDA things
if( SPIRIT_USE_CUDA )
    target_link_libraries( ${META_PROJECT_NAME} INTERFACE ${SPIRIT_CUDA_LIBS} )
    set_target_properties( ${META_PROJECT_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON )
    set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_CUDA )

    ### Need to use nvcc on many source files because they indirectly include cuda.h
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/src/data/Geometry.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/data/Spin_System.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/data/Spin_System_Chain.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Eigenmodes.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Hamiltonian.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Hamiltonian_Gaussian.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Sparse_HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Manifoldmath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method_MC.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method_LLG.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method_GNEB.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method_MMF.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Method_EMA.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Neighbours.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Solver_Kernels.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/engine/Vectormath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/Configwriter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/Configparser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/Dataparser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/Datawriter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/IO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/Filter_File_Handle.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/io/OVF_File.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Chain.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Configurations.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Geometry.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Hamiltonian.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/IO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Parameters_EMA.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Parameters_GNEB.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Parameters_LLG.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Parameters_MC.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Parameters_MMF.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Simulation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/State.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/System.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Transitions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/Spirit/Quantities.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/utility/Configurations.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/utility/Configuration_Chain.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/utility/Logging.cpp
        PROPERTIES LANGUAGE CUDA )

    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/test/test_api.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/test_io.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/test_physics.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/test_vectormath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/test_manifoldmath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/test/test_ema.cpp
        PROPERTIES LANGUAGE CUDA )
endif()

### Coverage flags and linking if needed
if( SPIRIT_BUILD_TEST AND SPIRIT_TEST_COVERAGE )
    set( CMAKE_CXX_FLAGS_COVERAGE
        "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline" )

    set_target_properties( ${META_PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS ${CMAKE_CXX_FLAGS_COVERAGE}
        LINK_FLAGS    ${CMAKE_CXX_FLAGS_COVERAGE} )

    target_link_libraries( ${META_PROJECT_NAME} INTERFACE
        ${CMAKE_CXX_FLAGS_COVERAGE} ${COVERAGE_LIBRARIES} )
endif()

### Definitions corresponding to build options
if( SPIRIT_ENABLE_DEFECTS )
    set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_ENABLE_DEFECTS )
endif()
if( SPIRIT_ENABLE_PINNING )
    set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_ENABLE_PINNING )
endif()
if( SPIRIT_USE_THREADS )
    set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DSPIRIT_USE_THREADS )
endif()

### Add compile definitions
target_compile_definitions( ${META_PROJECT_NAME} PRIVATE ${SPIRIT_COMPILE_DEFINITIONS} )
####################################################################


######### C/C++ and JavaScript #####################################
if( SPIRIT_BUILD_FOR_CXX OR SPIRIT_BUILD_FOR_JS )
    #--------------------------------------------------
    message( STATUS ">> Building static C/C++ library ${META_PROJECT_NAME}_static" )
    add_library( ${META_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
    ### Forward public includes from the object library
    target_include_directories( ${META_PROJECT_NAME}_static PUBLIC
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES> )
    ### Apply interface link dependencies from the object library
    target_link_libraries( ${META_PROJECT_NAME}_static PRIVATE
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_LINK_LIBRARIES> )
    #--------------------------------------------------

    #--------------------------------------------------
    if( SPIRIT_BUILD_TEST )
        message( STATUS ">> Building unit tests for Spirit C/C++ library" )

        #----- Enable CTest testing
        enable_testing()

        #----- Test creation macro
        macro( add_framework_test testName testSrc )
            # Executable
            add_executable( ${testName} test/main.cpp ${testSrc} )
            # Link to core library
            target_link_libraries( ${testName} ${META_PROJECT_NAME}_static )
            # Properties
            set_target_properties( ${testName} PROPERTIES
                CXX_STANDARD             14
                CXX_STANDARD_REQUIRED    ON
                CXX_EXTENSIONS           OFF
                RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY} )
            # Include Directories
            target_include_directories( ${testName} PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}/test
                ${CMAKE_CURRENT_LIST_DIR}/thirdparty
                ${FFT_INCLUDE_DIRS} )
            # Apply public includes from the object library
            target_include_directories( ${testName} PUBLIC
                $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
            # Apply the compile definitions, since some tests include "private" headers
            target_compile_definitions( ${testName} PRIVATE ${SPIRIT_COMPILE_DEFINITIONS} )
            # Coverage flags and linking if needed
            if( SPIRIT_BUILD_TEST AND SPIRIT_TEST_COVERAGE )
                set_property( TARGET ${testName} PROPERTY COMPILE_FLAGS ${CMAKE_CXX_FLAGS_COVERAGE} )
                set_property( TARGET ${testName} PROPERTY LINK_FLAGS    ${CMAKE_CXX_FLAGS_COVERAGE} )
            endif()
            # Add the test
            add_test( NAME        ${testName}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMAND           ${testName} --use-colour=yes )
            # Add to list
            set( TEST_EXECUTABLES ${TEST_EXECUTABLES} ${testName} )
        endmacro( add_framework_test testName testSrc )

        #----- Create test executables
        set( TEST_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
        set( TEST_EXECUTABLES )
        add_framework_test( test_vmath    test/test_vectormath.cpp )
        add_framework_test( test_mmath    test/test_manifoldmath.cpp )
        add_framework_test( test_api      test/test_api.cpp )
        add_framework_test( test_solvers  test/test_solvers.cpp )
        add_framework_test( test_physics  test/test_physics.cpp )
        add_framework_test( test_ema      test/test_ema.cpp)
        add_framework_test( test_io       test/test_io.cpp )
    endif()
    #--------------------------------------------------
endif()
####################################################################


######### Python ###################################################
if( SPIRIT_BUILD_FOR_PYTHON )
    #--------------------------------------------------
    message( STATUS ">> Building shared library for Python" )
    add_library( ${META_PROJECT_NAME}_python SHARED $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
    ### Forward public includes from the object library
    target_include_directories( ${META_PROJECT_NAME}_python PUBLIC
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES> )
    ### Apply interface link dependencies from the object library
    target_link_libraries( ${META_PROJECT_NAME}_python PRIVATE
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_LINK_LIBRARIES> )
    #--------------------------------------------------

    #--------------------------------------------------
    #----- Python module & library directories
    set( SPIRIT_PYTHON_MODULE_DIR ${CMAKE_CURRENT_LIST_DIR}/python )
    set( PYLIB_OUTPUT_DIR ${SPIRIT_PYTHON_MODULE_DIR}/spirit )

    #----- Utility python files
    configure_file( ${CMAKE_SOURCE_DIR}/LICENSE.txt ${SPIRIT_PYTHON_MODULE_DIR}/LICENSE.txt COPYONLY )
    configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMake/__init__.py.in ${PYLIB_OUTPUT_DIR}/__init__.py )

    file( WRITE ${PYLIB_OUTPUT_DIR}/scalar.py
        "import ctypes\n"
        "scalar = ctypes.c_${SPIRIT_SCALAR_TYPE}")

    file( WRITE ${PYLIB_OUTPUT_DIR}/version.py
        "version_major    = ${META_VERSION_MAJOR}\n"
        "version_minor    = ${META_VERSION_MINOR}\n"
        "version_patch    = ${META_VERSION_PATCH}\n"
        "version          = \"${META_VERSION}\"\n"
        "revision         = \"${META_VERSION_REVISION}\"\n"
        "version_full     = \"${META_VERSION} (${META_VERSION_REVISION})\"\n"
        "compiler         = \"${META_COMPILER}\"\n"
        "compiler_version = \"${META_COMPILER_VERSION}\"\n"
        "compiler_full    = \"${META_COMPILER_FULL}\"\n"
        "scalartype       = \"${SPIRIT_SCALAR_TYPE}\"\n"
        "pinning          = \"${SPIRIT_ENABLE_PINNING}\"\n"
        "defects          = \"${SPIRIT_ENABLE_DEFECTS}\"\n"
        "cuda             = \"${SPIRIT_USE_CUDA}\"\n"
        "openmp           = \"${SPIRIT_USE_OPENMP}\"\n"
        "threads          = \"${SPIRIT_USE_THREADS}\"\n"
        "fftw             = \"${SPIRIT_USE_FFTW}\"" )

    #----- Set output location & name
    ### We want it to be called spirit, not spirit_python, and to
    ### be placed under python/spirit/ s.t. it is directly part of
    ### the python spirit bindings module/package
    set_target_properties( ${META_PROJECT_NAME}_python PROPERTIES
        OUTPUT_NAME "${META_PROJECT_NAME}"
        LIBRARY_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR} )
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        set_target_properties( ${META_PROJECT_NAME}_python PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PYLIB_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PYLIB_OUTPUT_DIR} )
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
    #--------------------------------------------------

    #--------------------------------------------------
    if( SPIRIT_BUILD_TEST )
        message( STATUS ">> Setting up unit tests for python bindings" )
        # find_package( PythonInterp 2.7 REQUIRED )
        find_package( PythonInterp REQUIRED )

        #----- Enable CTest testing
        enable_testing()

        #----- Test creation macro
        macro( add_python_test test_name src )
            # Add the test
            add_test(
                NAME                ${test_name}
                WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
                COMMAND             ${PYTHON_EXECUTABLE} "${SPIRIT_PYTHON_MODULE_DIR}/test/${src}" )
            # Properties
            set_tests_properties( ${test_name}
                PROPERTIES ENVIRONMENT "PYTHONPATH=${SPIRIT_PYTHON_MODULE_DIR}:$PYTHONPATH" )
            # Add to list
            set( PYTHON_TEST_EXECUTABLES ${PYTHON_TEST_EXECUTABLES} ${test_name} )
        endmacro( add_python_test )

        #----- Create test executables
        set( PYTHON_TEST_EXECUTABLES )
        add_python_test( test_python_chain           chain.py )
        add_python_test( test_python_configuration   configuration.py )
        add_python_test( test_python_constants       constants.py )
        add_python_test( test_python_geometry        geometry.py )
        add_python_test( test_python_hamiltonian     hamiltonian.py )
        add_python_test( test_python_io              io_test.py )
        add_python_test( test_python_log             log.py )
        add_python_test( test_python_parameters      parameters.py )
        add_python_test( test_python_quantities      quantities.py )
        add_python_test( test_python_simulation      simulation.py )
        add_python_test( test_python_state           state.py )
        add_python_test( test_python_system          system.py )
        add_python_test( test_python_transition      transition.py )
    endif()
    #--------------------------------------------------
endif()
####################################################################


######### Julia ####################################################
if( SPIRIT_BUILD_FOR_JULIA )
    message( STATUS ">> Building shared library for Julia" )

    #--------------------------------------------------
    add_library( ${META_PROJECT_NAME}_julia SHARED $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
    ### Forward public includes from the object library
    target_include_directories(${META_PROJECT_NAME}_julia PUBLIC
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
    ### Apply interface link dependencies from the object library
    target_link_libraries(${META_PROJECT_NAME}_julia PRIVATE
        $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_LINK_LIBRARIES>)
    #--------------------------------------------------

    #----- Set output location & name
    ### We want it to be called spirit, not spirit_julia, and to
    ### be placed under julia/spirit/ s.t. it is directly part of
    ### the julia spirit bindings module/package
    set( JLLIB_OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/julia/Spirit/ )
    set_target_properties( ${META_PROJECT_NAME}_julia PROPERTIES
        OUTPUT_NAME              "${META_PROJECT_NAME}"
        LIBRARY_OUTPUT_DIRECTORY ${JLLIB_OUTPUT_DIR} )
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        set_target_properties( ${META_PROJECT_NAME}_julia PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${JLLIB_OUTPUT_DIR} )
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
    #--------------------------------------------------
endif()
####################################################################


######### IDE Folders ##############################################
### Folder include
source_group( "include"          FILES ${HEADER_SPIRIT_ROOT} )
source_group( "include\\data"    FILES ${HEADER_SPIRIT_DATA} )
source_group( "include\\engine"  FILES ${HEADER_SPIRIT_ENGINE} )
source_group( "include\\io"      FILES ${HEADER_SPIRIT_IO} )
source_group( "include\\utility" FILES ${HEADER_SPIRIT_UTILITY} )
source_group( "include\\spirit"  FILES ${HEADER_SPIRIT} )
### Folder src
source_group( "src"          FILES ${SOURCE_SPIRIT_ROOT} ) #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
source_group( "src\\data"    FILES ${SOURCE_SPIRIT_DATA} )
source_group( "src\\engine"  FILES ${SOURCE_SPIRIT_ENGINE} )
source_group( "src\\io"      FILES ${SOURCE_SPIRIT_IO} )
source_group( "src\\utility" FILES ${SOURCE_SPIRIT_UTILITY} )
source_group( "src\\spirit"  FILES ${SOURCE_SPIRIT} )
####################################################################


######### Install ##################################################
#----- Documentation
install( DIRECTORY docs/ DESTINATION docs/Spirit/core/ COMPONENT spirit_core_documentation )
#--------------------------------------------------

#----- Include folder
install( DIRECTORY include/Spirit DESTINATION include COMPONENT spirit_core_headers
    FILES_MATCHING PATTERN "*.h" )
#--------------------------------------------------

#----- Static library and tests
if( SPIRIT_BUILD_FOR_CXX )
    install( TARGETS ${META_PROJECT_NAME}_static DESTINATION lib COMPONENT spirit_core_archives )
    if( SPIRIT_BUILD_TEST )
        install( TARGETS ${TEST_EXECUTABLES} DESTINATION test COMPONENT spirit_core_tests )
        install( DIRECTORY test/input DESTINATION test COMPONENT spirit_core_tests )
    endif()
endif()
#--------------------------------------------------

#----- Shared library and python tests
if( SPIRIT_BUILD_FOR_PYTHON )
    install( TARGETS ${META_PROJECT_NAME}_python DESTINATION bin COMPONENT spirit_core_python )
    install( DIRECTORY python/spirit DESTINATION python COMPONENT spirit_core_python
        FILES_MATCHING PATTERN "*.py" )
    if( SPIRIT_BUILD_TEST )
        # install( TARGETS ${PYTHON_TEST_EXECUTABLES} DESTINATION python/test )
        install( DIRECTORY python/test/ DESTINATION test/python COMPONENT spirit_core_python
            FILES_MATCHING PATTERN "*.py" )
    endif()
endif()
#--------------------------------------------------

# #----- Shared library and julia tests
# if( SPIRIT_BUILD_FOR_JULIA )
#     install( TARGETS   ${META_PROJECT_NAME}_julia DESTINATION bin )
#     install( DIRECTORY julia/Spirit               DESTINATION julia )
#     # if( SPIRIT_BUILD_TEST )
#     #   install( TARGETS ${JULIA_TEST_EXECUTABLES} DESTINATION julia/test )
#     # endif()
# endif()
# #--------------------------------------------------

####################################################################


######### Export ####################################################
### Library
set( SPIRIT_OBJS             $<TARGET_OBJECTS:${META_PROJECT_NAME}>  PARENT_SCOPE )
set( SPIRIT_LIBRARIES        ${META_PROJECT_NAME}                    PARENT_SCOPE )
set( SPIRIT_LIBRARIES_STATIC ${META_PROJECT_NAME}_static             PARENT_SCOPE )
set( SPIRIT_LINK_DEPS        ${qhull_LIBS} ${OVF_LIBRARIES_STATIC} ${FFT_LIB} ${THREAD_LIBS} PARENT_SCOPE )
set( SPIRIT_INCLUDE_DIRS     ${SPIRIT_INCLUDE_DIRS}                  PARENT_SCOPE )
### Meta information
set( META_PROJECT_NAME        ${META_PROJECT_NAME}        PARENT_SCOPE )
set( META_PROJECT_DESCRIPTION ${META_PROJECT_DESCRIPTION} PARENT_SCOPE )
set( META_AUTHOR_ORGANIZATION ${META_AUTHOR_ORGANIZATION} PARENT_SCOPE )
set( META_AUTHOR_DOMAIN       ${META_AUTHOR_DOMAIN}       PARENT_SCOPE )
set( META_AUTHOR_MAINTAINER   ${META_AUTHOR_MAINTAINER}   PARENT_SCOPE )
set( META_AUTHOR_EMAIL        ${META_AUTHOR_EMAIL}        PARENT_SCOPE )
set( META_VERSION             ${META_VERSION}             PARENT_SCOPE )
set( META_VERSION_REVISION    ${META_VERSION_REVISION}    PARENT_SCOPE )
####################################################################


if( SPIRIT_PRINT_SOURCES )
    MESSAGE( STATUS ">> Headers:                    ${HEADER_SPIRIT_ROOT} ${HEADER_SPIRIT_DATA} ${HEADER_SPIRIT_ENGINE} ${HEADER_SPIRIT} ${HEADER_SPIRIT_UTILITY}" )
    MESSAGE( STATUS ">> Sources:                    ${SOURCE_SPIRIT_ROOT} ${SOURCE_SPIRIT_DATA} ${SOURCE_SPIRIT_ENGINE} ${SOURCE_SPIRIT} ${SOURCE_SPIRIT_UTILITY}" )
endif()
MESSAGE( STATUS ">> --------------------- Spirit done ---------------------------------- <<" )
message( STATUS ">> CMake CXX Flags:        ${CMAKE_CXX_FLAGS}" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )