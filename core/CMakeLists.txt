MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Spirit --------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#############################################


######### Build Flags #######################
### CMake Verbosity
option( SPIRIT_PRINT_SOURCES     "Print Headers and Sources from Cmake."          OFF )
### These decide which projects are built
option( SPIRIT_BUILD_FOR_JS      "Build the JavaScript library."                  OFF )
option( SPIRIT_BUILD_FOR_JULIA   "Build the shared library for Julia."            OFF )
option( SPIRIT_BUILD_FOR_PYTHON  "Build the shared library for Python."           ON  )
option( SPIRIT_BUILD_FOR_CXX     "Build the static library for C++ applications"  ON  )
### Feature switches for Spirit
option( SPIRIT_ENABLE_PINNING    "Enable pinning individual or rows of spins."    OFF )
option( SPIRIT_ENABLE_DEFECTS    "Enable defects and disorder in the lattice."    OFF )
### Options for Spirit
option( SPIRIT_BUILD_TEST        "Build unit tests for the Spirit library."                ON  )
option( SPIRIT_USE_CUDA          "Use CUDA to speed up certain parts of the code."         OFF )
option( SPIRIT_USE_THREADS       "Use std threads to speed up certain parts of the code."  OFF )
### Set the scalar type used in the Spirit library
set( SPIRIT_SCALAR_TYPE double )
#############################################


##### Some restrictions on the options ######
if( SPIRIT_USE_CUDA )
	### If cuda is used, we use float and
	### we cannot build for JS or Julia
	set( SPIRIT_SCALAR_TYPE        float )
	set( SPIRIT_BUILD_FOR_JS        OFF )
	set( SPIRIT_BUILD_FOR_JULIA     OFF )
endif()
#############################################
if( SPIRIT_BUILD_FOR_JS )
	### UI-Web needs float
	set( SPIRIT_SCALAR_TYPE 	float )
	### UI-Web needs to be built alone, as it
	### uses a different toolchain
	set( SPIRIT_BUILD_TEST          OFF )
	set( SPIRIT_BUILD_FOR_JULIA 	OFF )
	set( SPIRIT_BUILD_FOR_PYTHON 	OFF )
	set( SPIRIT_BUILD_FOR_CXX 		OFF )
	### Emscripten cannot use cuda or threads
	set( SPIRIT_USE_CUDA 		OFF )
	set( SPIRIT_USE_THREADS 	OFF )
endif()
#############################################
if( SPIRIT_BUILD_TEST )
    enable_testing()
endif()
#############################################


### Set the cmake subdirectory
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )


######### Get git revision ##################
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(SUBSTRING "${GIT_SHA1}" 0 12 GIT_REV)
#############################################


######### Meta information about the project
set(META_PROJECT_NAME         "Spirit")
set(META_PROJECT_DESCRIPTION  "Optimizations and Dynamics Framework for atomistic Spin systems")
set(META_AUTHOR_ORGANIZATION  "")
set(META_AUTHOR_DOMAIN        "https://spirit-code.github.io")
set(META_AUTHOR_MAINTAINER    "Gideon Mueller")
set(META_AUTHOR_EMAIL         "g.mueller@fz-juelich.de")
set(META_VERSION_MAJOR        "1")
set(META_VERSION_MINOR        "6")
set(META_VERSION_PATCH        "1")
set(META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}")
set(META_VERSION_REVISION     "${GIT_REV}")
### Propagate version and name_version upwards
set(SPIRIT_META_VERSION              "${META_VERSION}" PARENT_SCOPE)
set(SPIRIT_META_NAME_VERSION         "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})" PARENT_SCOPE)
#############################################

######### Project Name ######################
project(${META_PROJECT_NAME} VERSION ${SPIRIT_META_VERSION})
#############################################


######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated Spirit_Defines.h
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### QHull external Project ############
if((NOT qhull_LIBS) OR (NOT qhull_INCLUDE_DIRS))

	if(qhull_LIBS)
		message(WARNING "qhull_LIBS is set, but qhull_INCLUDE_DIRS is missing.")
	endif()
	if(qhull_INCLUDE_DIRS)
		message(WARNING "qhull_INCLUDE_DIRS is set, but qhull_LIBS is missing.")
	endif()

	include(ExternalProject)
	ExternalProject_add(qhull
		GIT_REPOSITORY https://github.com/qhull/qhull.git
		CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/thirdparty-install"
		PREFIX qhull-prefix
	)

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
	set(qhull_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/thirdparty-install/include;${PROJECT_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
endif()
#############################################


######### Generate Spirit_Defines.h ################
set(VERSION_STRING "${META_VERSION}")
set(VERSION_REVISION_STRING "${META_VERSION_REVISION}")
string(TOUPPER ${SPIRIT_SCALAR_TYPE} SPIRIT_SCALAR_TYPE_UPPERCASE)
if ( SPIRIT_ENABLE_DEFECTS )
	set ( SPIRIT_DEFINE_DEFECTS "#define SPIRIT_ENABLE_DEFECTS")
endif()
if ( SPIRIT_ENABLE_PINNING )
	set ( SPIRIT_DEFINE_PINNING "#define SPIRIT_ENABLE_PINNING")
endif()
if ( SPIRIT_USE_CUDA )
	set ( SPIRIT_DEFINE_CUDA "#define SPIRIT_USE_CUDA")
endif()
if ( SPIRIT_USE_THREADS )
	set ( SPIRIT_DEFINE_THREADS "#define SPIRIT_USE_THREADS")
endif()
configure_file(${PROJECT_SOURCE_DIR}/CMake/Spirit_Defines.h.in ${PROJECT_SOURCE_DIR}/include/Spirit_Defines.h)
#############################################


######### CUDA decisions ####################
if ( SPIRIT_USE_CUDA )
    find_package( CUDA REQUIRED )
    add_definitions( -DUSE_CUDA )
	# set (CUDA_PROPAGATE_HOST_FLAGS ON)
	# --std=c++11 flag may be necessary, but it is propagated from global flags...
	# if it appears twice, for some reason the compilation breaks
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 --expt-relaxed-constexpr" )
endif()
#############################################


######### Where to search for library headers
set( EIGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty )
set( SPECTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty ${PROJECT_SOURCE_DIR}/thirdparty/spectra/include )
set( SPIRIT_INCLUDE_DIRS
	 ${PROJECT_SOURCE_DIR}/include
	 ${PROJECT_SOURCE_DIR}/include/data
	 ${PROJECT_SOURCE_DIR}/include/engine
	 ${PROJECT_SOURCE_DIR}/include/Spirit
	 ${PROJECT_SOURCE_DIR}/include/utility 
	 ${PROJECT_SOURCE_DIR}/src/utility
	 ${EIGEN_INCLUDE_DIRS}
	 ${SPECTRA_INCLUDE_DIRS}
	 ${qhull_INCLUDE_DIRS} )
include_directories( ${SPIRIT_INCLUDE_DIRS} )
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_SPIRIT_ROOT)
set(HEADER_SPIRIT)
set(HEADER_SPIRIT_DATA)
set(HEADER_SPIRIT_ENGINE)
set(HEADER_SPIRIT_UTILITY)
### Source Groups
set(SOURCE_SPIRIT_ROOT)
set(SOURCE_SPIRIT)
set(SOURCE_SPIRIT_DATA)
set(SOURCE_SPIRIT_ENGINE)
set(SOURCE_SPIRIT_UTILITY)
#############################################


######### Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/include)
#############################################


######### IDE Folders #######################
### Folder include
source_group("include" FILES ${HEADER_SPIRIT_ROOT})
source_group("include\\data" FILES ${HEADER_SPIRIT_DATA})
source_group("include\\engine" FILES ${HEADER_SPIRIT_ENGINE})
source_group("include\\utility" FILES ${HEADER_SPIRIT_UTILITY})
source_group("include\\spirit" FILES ${HEADER_SPIRIT})
### Folder src
source_group("src" FILES  ${SOURCE_SPIRIT_ROOT}) #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
source_group("src\\data" FILES ${SOURCE_SPIRIT_DATA})
source_group("src\\engine" FILES ${SOURCE_SPIRIT_ENGINE})
source_group("src\\utility" FILES ${SOURCE_SPIRIT_UTILITY})
source_group("src\\spirit" FILES ${SOURCE_SPIRIT})
#############################################


#############################################
set( SPIRIT_LIBRARY_SOURCES 
	${HEADER_SPIRIT_ROOT}
    ${HEADER_SPIRIT_DATA}
    ${HEADER_SPIRIT_ENGINE}
	${HEADER_SPIRIT}
	${HEADER_SPIRIT_UTILITY}
	${SOURCE_SPIRIT_ROOT}
	${SOURCE_SPIRIT_DATA}
	${SOURCE_SPIRIT_ENGINE}
	${SOURCE_SPIRIT}
	${SOURCE_SPIRIT_UTILITY} )

if( NOT SPIRIT_USE_CUDA )
	######### Tell CMake to create the object files
	add_library( ${META_PROJECT_NAME} OBJECT ${SPIRIT_LIBRARY_SOURCES} )
	set_property(TARGET ${META_PROJECT_NAME} PROPERTY CXX_STANDARD 11)
	set_property(TARGET ${META_PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET ${META_PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
	add_dependencies(${META_PROJECT_NAME} ${qhull_LIBS})
	######### Tell CMake to create the static spirit library for the QT UI
	if( SPIRIT_BUILD_FOR_CXX OR SPIRIT_BUILD_FOR_JS)
		MESSAGE( STATUS ">> Building static cxx library ${META_PROJECT_NAME}_static" )
		add_library( ${META_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
    	target_include_directories( ${META_PROJECT_NAME}_static PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)
		target_link_libraries( ${META_PROJECT_NAME}_static ${qhull_LIBS} )
		# set_target_properties( ${META_PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${META_PROJECT_NAME}" )
	endif()
else()
	include_directories( ${META_PROJECT_NAME}_static PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
	if( SPIRIT_BUILD_FOR_CXX )
		cuda_add_library( ${META_PROJECT_NAME}_static STATIC ${SPIRIT_LIBRARY_SOURCES} )
		add_dependencies(${META_PROJECT_NAME}_static ${qhull_LIBS})
		target_link_libraries( ${META_PROJECT_NAME}_static ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
endif()
#############################################

######### Tell CMake to create the shared library for the Python UI
if( SPIRIT_BUILD_FOR_PYTHON )
	set( SPIRIT_PYDIR ${PROJECT_SOURCE_DIR}/python )
	set( PYLIB_OUTPUT_DIR ${SPIRIT_PYDIR}/spirit )
	### Utility python files
	configure_file(${CMAKE_SOURCE_DIR}/LICENSE.txt ${SPIRIT_PYDIR}/LICENSE.txt COPYONLY)
	configure_file(${PROJECT_SOURCE_DIR}/CMake/__init__.py.in ${PYLIB_OUTPUT_DIR}/__init__.py)
	file( WRITE ${PYLIB_OUTPUT_DIR}/scalar.py "import ctypes\nscalar=ctypes.c_${SPIRIT_SCALAR_TYPE}")
	file( WRITE ${PYLIB_OUTPUT_DIR}/version.py "version=\"${META_VERSION}\"")
	###
	if( NOT SPIRIT_USE_CUDA )
		MESSAGE( STATUS ">> Building shared library for Python" )
		#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
		add_library( ${META_PROJECT_NAME}_python SHARED $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
		target_link_libraries( ${META_PROJECT_NAME}_python ${qhull_LIBS} )
	else()
		MESSAGE( STATUS ">> Building shared CUDA library for Python" )
		include_directories( ${META_PROJECT_NAME}_python PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
		cuda_add_library( ${META_PROJECT_NAME}_python SHARED ${SPIRIT_LIBRARY_SOURCES} )
		add_dependencies(${META_PROJECT_NAME}_static ${qhull_LIBS})
		target_link_libraries( ${META_PROJECT_NAME}_python ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
	### We want it to be called spirit, not spirit_python
	set_target_properties( ${META_PROJECT_NAME}_python PROPERTIES OUTPUT_NAME "${META_PROJECT_NAME}" )
	### We want it to be placed under python/Spirit/ s.t. it is directly part of the python Spirit bindings module/package
	set_target_properties( ${META_PROJECT_NAME}_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR} )
	if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
			set_target_properties( ${META_PROJECT_NAME}_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${PYLIB_OUTPUT_DIR} )
		endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
	endif ( MSVC )
endif()
#############################################

######### Tell CMake to create the shared library for the Julia UI
if( SPIRIT_BUILD_FOR_JULIA )
	MESSAGE( STATUS ">> Building shared library for Julia" )
	#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
	add_library( ${META_PROJECT_NAME}_julia SHARED $<TARGET_OBJECTS:${META_PROJECT_NAME}> )
	target_link_libraries( ${META_PROJECT_NAME}_julia ${qhull_LIBS})
	### We want it to be called ${META_PROJECT_NAME}, not ${META_PROJECT_NAME}_julia
	set_target_properties( ${META_PROJECT_NAME}_julia PROPERTIES OUTPUT_NAME "spirit" )
	### We want it to be placed under julia/Spirit/ s.t. it is directly part of the julia spirit bindings module/package
	set( JLLIB_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/julia/Spirit/ )
    set_target_properties( ${META_PROJECT_NAME}_julia PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${JLLIB_OUTPUT_DIR} )
    if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        	set_target_properties( ${META_PROJECT_NAME}_julia PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${JLLIB_OUTPUT_DIR} )
		endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
    endif ( MSVC )
endif()
#############################################



######### Test executable ###################
set( TEST_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR} )
### Test creation macro
macro( add_framework_test testName testSrc )
    # Executable
    add_executable( ${testName} test/main.cpp ${testSrc} )
    # Link Library
    target_link_libraries( ${testName} ${META_PROJECT_NAME}_static )
    # Properties
    set_property(TARGET ${testName} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${testName} PROPERTY CXX_EXTENSIONS OFF)
    # Include Directories
    target_include_directories( ${testName} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND           ${testName} --use-colour=yes )
endmacro( add_framework_test testName testSrc )
### Create tests if needed
if ( SPIRIT_BUILD_TEST AND SPIRIT_BUILD_FOR_CXX )
	MESSAGE( STATUS ">> Building unit tests for Spirit" )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    add_framework_test( test_vmath test/test_vectormath.cpp )
    add_framework_test( test_mmath test/test_manifoldmath.cpp )
    add_framework_test( test_api   test/test_api.cpp )
    add_framework_test( test_optim test/test_optimizers.cpp )
endif()
#############################################


######### Python Test #######################
macro(add_python_test test_name src)
	# Add the test
	add_test(
		NAME                ${test_name}
		WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
		COMMAND             ${PYTHON_EXECUTABLE} "${SPIRIT_PYDIR}/test/${src}")
	# Properties
	set_tests_properties(${test_name}
		PROPERTIES ENVIRONMENT "PYTHONPATH=${SPIRIT_PYDIR}:$PYTHONPATH")
endmacro(add_python_test)
### Create tests if needed
if( SPIRIT_BUILD_TEST AND SPIRIT_BUILD_FOR_PYTHON )
	MESSAGE( STATUS ">> Setting up unit tests for python bindings" )
	# find_package( PythonInterp 2.7 REQUIRED )
	find_package( PythonInterp REQUIRED )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    add_python_test( test_python_chain           chain.py )
    add_python_test( test_python_configuration   configuration.py )
    add_python_test( test_python_constants       constants.py )
    add_python_test( test_python_geometry        geometry.py )
    add_python_test( test_python_hamiltonian     hamiltonian.py )
    add_python_test( test_python_io              io.py )
    add_python_test( test_python_log             log.py )
    add_python_test( test_python_parameters      parameters.py )
    add_python_test( test_python_quantities      quantities.py )
    add_python_test( test_python_simulation      simulation.py )
    add_python_test( test_python_state           state.py )
    add_python_test( test_python_system          system.py )
    add_python_test( test_python_transition      transition.py )
endif( )
#############################################


######### Export
set(SPIRIT_OBJS             $<TARGET_OBJECTS:${META_PROJECT_NAME}> PARENT_SCOPE)
set(SPIRIT_LIBRARIES        ${META_PROJECT_NAME}        PARENT_SCOPE)
set(SPIRIT_LIBRARIES_STATIC ${META_PROJECT_NAME}_static	PARENT_SCOPE)
set(SPIRIT_INCLUDE_DIRS     ${SPIRIT_INCLUDE_DIRS}      PARENT_SCOPE)
set(HEADER_SPIRIT_DATA      ${HEADER_SPIRIT_DATA}       PARENT_SCOPE)
set(HEADER_SPIRIT_ENGINE    ${HEADER_SPIRIT_ENGINE}     PARENT_SCOPE)
set(HEADER_SPIRIT 			${HEADER_SPIRIT} 		    PARENT_SCOPE)
set(HEADER_SPIRIT_UTILITY   ${HEADER_SPIRIT_UTILITY}    PARENT_SCOPE)
set(SOURCE_SPIRIT_DATA      ${SOURCE_SPIRIT_DATA}       PARENT_SCOPE)
set(SOURCE_SPIRIT_ENGINE    ${SOURCE_SPIRIT_ENGINE}     PARENT_SCOPE)
set(SOURCE_SPIRIT 			${SOURCE_SPIRIT} 		    PARENT_SCOPE)
set(SOURCE_SPIRIT_UTILITY   ${SOURCE_SPIRIT_UTILITY}    PARENT_SCOPE)
#############################################



######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    ${HEADER_SPIRIT_ROOT} ${HEADER_SPIRIT_DATA} ${HEADER_SPIRIT_ENGINE} ${HEADER_SPIRIT} ${HEADER_SPIRIT_UTILITY}" )
	MESSAGE( STATUS ">> Sources:                    ${SOURCE_SPIRIT_ROOT} ${SOURCE_SPIRIT_DATA} ${SOURCE_SPIRIT_ENGINE} ${SOURCE_SPIRIT} ${SOURCE_SPIRIT_UTILITY}" )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- Spirit done ---------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )