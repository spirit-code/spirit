cmake_minimum_required( VERSION 3.5 FATAL_ERROR )

######### Meta information about the project
set( META_PROJECT_NAME         "ovf" )
set( META_PROJECT_DESCRIPTION  "OVF file parser" )
set( META_AUTHOR_DOMAIN        "https://github.com/spirit-code/ovf" )
set( META_AUTHOR_MAINTAINER    "Gideon Mueller" )
set( META_AUTHOR_EMAIL         "g.mueller@fz-juelich.de" )
set( META_VERSION_MAJOR        "0" )
set( META_VERSION_MINOR        "4" )
set( META_VERSION_PATCH        "1" )
set( META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}" )
#############################################

project( ${META_PROJECT_NAME} VERSION ${META_VERSION} )

set( CMAKE_POSITION_INDEPENDENT_CODE ON )

#############################################
option( OVF_BUILD_PYTHON_BINDINGS  "Build the shared library for Python."  OFF )
option( OVF_BUILD_FORTRAN_BINDINGS "Build the module file for Fortran."    OFF )
option( OVF_BUILD_TEST             "Build unit tests for the ovf library." OFF )
#############################################

set( SOURCE_FILES src/ovf.cpp )

set( HEADER_FILES
    src/ovf.h
    src/detail/helpers.hpp
    src/detail/parse.hpp
    src/detail/parse_rules.hpp
    src/detail/write.hpp
)

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" )
    add_definitions( -DNOMINMAX )
endif()

### flag for fmt to be header-only
add_definitions( -DFMT_HEADER_ONLY -DFMT_TEST=no )


######## GNU Compiler Collection - gcc ###############################
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    # Require at least gcc 7
    if( OVF_BUILD_FORTRAN_BINDINGS AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7 )
        message( FATAL_ERROR "GCC version must be at least 7! Older versions contain incompatible bugs for Fortran..." )
    endif()
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3" )
    ### Linker Flags
    if( APPLE )
        set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind" )
    endif()
######## Apple Clang #################################################
elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3" )
    ### Linker Flags
    if( APPLE )
        set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind" )
    endif()
######## Intel Compiler - icc ########################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3" )
    ### Linker Flags
    if( APPLE )
        set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind" )
    endif()
endif()
######################################################################


### Object library
add_library( ${PROJECT_NAME} OBJECT ${SOURCE_FILES} )

set_property( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11 )
set_property( TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF )

target_include_directories( ${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_include_directories( ${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/PEGTL/include )

### Static library
add_library( ${PROJECT_NAME}_static $<TARGET_OBJECTS:${PROJECT_NAME}> )

install( TARGETS ${PROJECT_NAME}_static DESTINATION lib )
install( DIRECTORY include/ovf DESTINATION include )


### Build Python bindings
if( OVF_BUILD_PYTHON_BINDINGS )
    add_library( ${PROJECT_NAME}_python SHARED $<TARGET_OBJECTS:${PROJECT_NAME}> )

    set_property( TARGET ${PROJECT_NAME}_python PROPERTY OUTPUT_NAME "${PROJECT_NAME}" )
    set_property( TARGET ${PROJECT_NAME}_python PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ovf )

    ### Utility python files
    configure_file( ${PROJECT_SOURCE_DIR}/LICENSE.txt ${PROJECT_SOURCE_DIR}/python/LICENSE.txt COPYONLY )
    configure_file( ${PROJECT_SOURCE_DIR}/CMake/__init__.py.in ${PROJECT_SOURCE_DIR}/python/ovf/__init__.py )
endif()


### Build Fortran bindings
if( OVF_BUILD_FORTRAN_BINDINGS )
    enable_language( Fortran )
    ######## GNU Compiler Collection - gcc ###############################
    if( "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" )
        set( CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -g3 -fcheck=all" )
    ######## flang #######################################################
    elseif( "${CMAKE_Fortran_COMPILER_ID}" MATCHES "Clang" )
        set( CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -g3 -fcheck=all" )
    ######## Intel Compiler - icc ########################################
    elseif ("${CMAKE_Fortran_COMPILER_ID}" MATCHES "Intel")
        set( CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS} -g3 -traceback" )
    endif()

    # this is where we will place the Fortran module files
    set( CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_SOURCE_DIR}/fortran )

    add_library( ${PROJECT_NAME}_fortran STATIC ${PROJECT_SOURCE_DIR}/fortran/ovf.f90 )
    target_link_libraries( ${PROJECT_NAME}_fortran ${PROJECT_NAME}_static )

    # set_property(TARGET ${PROJECT_NAME}_fortran PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/fortran)
endif()


set( TEST_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
set( TEST_EXECUTABLES )


### CXX test creation macro
macro( add_cxx_test testName testSrc )
    # Executable
    add_executable( ${testName} test/main.cpp ${PROJECT_SOURCE_DIR}/test/${testSrc} )
    # Link Library
    target_link_libraries( ${testName} ${PROJECT_NAME}_static )
    # Properties
    set_property( TARGET ${testName} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY} )
    set_property( TARGET ${testName} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${testName} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${testName} PROPERTY CXX_EXTENSIONS OFF )
    # Include Directories
    target_include_directories( ${testName} PRIVATE ${PROJECT_SOURCE_DIR}/test )
    target_include_directories( ${testName} PRIVATE ${PROJECT_SOURCE_DIR}/include )
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND           ${testName} --use-colour=yes )
    # Add to list
    set( TEST_EXECUTABLES ${TEST_EXECUTABLES} ${testName} )
endmacro( add_cxx_test testName testSrc )
###
if ( OVF_BUILD_TEST )
    MESSAGE( STATUS ">> Building unit tests for Spirit" )
    ### Enable CTest testing
    enable_testing()
    ### Tests
    add_cxx_test( test_cpp_simple simple.cpp )
    add_cxx_test( test_cpp_binary binary.cpp )
endif()


### Python test creation macro
set( PYTHON_TEST_EXECUTABLES )
macro( add_python_test test_name src )
    # Add the test
    add_test(
        NAME                ${test_name}
        WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
        COMMAND             ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/python/test/${src}" )
    # Properties
    set_tests_properties( ${test_name}
        PROPERTIES ENVIRONMENT "PYTHONPATH=${SPIRIT_PYDIR}:$PYTHONPATH" )
    # Add to list
    set( PYTHON_TEST_EXECUTABLES ${PYTHON_TEST_EXECUTABLES} ${test_name} )
endmacro( add_python_test )
###
if( OVF_BUILD_TEST AND OVF_BUILD_PYTHON_BINDINGS )
    MESSAGE( STATUS ">> Setting up unit tests for python bindings" )
    # find_package( PythonInterp 2.7 REQUIRED )
    find_package( PythonInterp REQUIRED )
    ### Tests
    add_python_test( test_python_simple simple.py )
    add_python_test( test_python_binary binary.py )
endif()


### Fortran test creation macro
macro( add_fortran_test testName testSrc )
    # Executable
    add_executable( ${testName} ${PROJECT_SOURCE_DIR}/fortran/test/${testSrc} )
    # Link Library
    target_link_libraries( ${testName} ${PROJECT_NAME}_fortran )
    # Properties
    set_property( TARGET ${testName} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY} )
    set_property( TARGET ${testName} PROPERTY LINKER_LANGUAGE Fortran )
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND           ${testName} --use-colour=yes )
    # Add to list
    set( TEST_EXECUTABLES ${TEST_EXECUTABLES} ${testName} )
endmacro( add_fortran_test testName testSrc )
###
if( OVF_BUILD_TEST AND OVF_BUILD_FORTRAN_BINDINGS )
    MESSAGE( STATUS ">> Setting up unit tests for fortran bindings" )
    ### Tests
    add_fortran_test( test_fortran_simple simple.f90 )
endif()


set( OVF_LIBRARIES_OBJECT ${PROJECT_NAME}        PARENT_SCOPE )
set( OVF_LIBRARIES_STATIC ${PROJECT_NAME}_static PARENT_SCOPE )