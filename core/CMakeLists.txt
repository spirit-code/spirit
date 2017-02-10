MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Core ----------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#############################################


######### Project Name ######################
project(SpiritCore VERSION ${META_VERSION})
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
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated Core_Defines.h
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
	set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
	add_dependencies(libqhullstatic_r qhull)

	add_library(libqhullcpp STATIC IMPORTED)
	set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
	set_property(TARGET libqhullcpp PROPERTY INTERFACE_LINK_LIBRARIES libqhullstatic_r)
	add_dependencies(libqhullcpp qhull)

	set(qhull_LIBS libqhullcpp)
	set(qhull_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/thirdparty-install/include;${PROJECT_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
endif()
#############################################


######### Generate Core_Defines.h ################
set(VERSION_STRING "${META_VERSION}")
set(VERSION_REVISION_STRING "${META_VERSION_REVISION}")
string(TOUPPER ${CORE_SCALAR_TYPE} CORE_SCALAR_TYPE_UPPERCASE)
if ( CORE_USE_CUDA )
	set ( CORE_DEFINE_CUDA "#define CORE_USE_CUDA")
endif()
if ( CORE_USE_THREADS )
	set ( CORE_DEFINE_THREADS "#define CORE_USE_THREADS")
endif()
configure_file(${CMAKE_SOURCE_DIR}/CMake/Core_Defines.h.in ${PROJECT_SOURCE_DIR}/include/Core_Defines.h)
#############################################


######### CUDA decisions ####################
if ( CORE_USE_CUDA )
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
set( CORE_INCLUDE_DIRS
	 ${PROJECT_SOURCE_DIR}/include
	 ${PROJECT_SOURCE_DIR}/include/data
	 ${PROJECT_SOURCE_DIR}/include/engine
	 ${PROJECT_SOURCE_DIR}/include/interface
	 ${PROJECT_SOURCE_DIR}/include/utility 
	 ${PROJECT_SOURCE_DIR}/src/utility )
include_directories( ${CORE_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${SPECTRA_INCLUDE_DIRS} ${QHULL_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_CORE_ROOT)
set(HEADER_CORE_ENGINE)
set(HEADER_CORE_UTILITY)
### Source Groups
set(SOURCE_CORE_ROOT)
set(SOURCE_CORE_ENGINE)
set(SOURCE_CORE_UTILITY)
#############################################


######### Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/include)
#############################################


######### IDE Folders #######################
### Folder include
source_group("include" FILES ${HEADER_CORE_ROOT})
source_group("include\\data" FILES ${HEADER_CORE_DATA})
source_group("include\\engine" FILES ${HEADER_CORE_ENGINE})
source_group("include\\utility" FILES ${HEADER_CORE_UTILITY})
source_group("include\\interface" FILES ${HEADER_CORE_INTERFACE})
### Folder src
source_group("src" FILES  ${SOURCE_CORE_ROOT}) #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
source_group("src\\data" FILES ${SOURCE_CORE_DATA})
source_group("src\\engine" FILES ${SOURCE_CORE_ENGINE})
source_group("src\\utility" FILES ${SOURCE_CORE_UTILITY})
source_group("src\\interface" FILES ${SOURCE_CORE_INTERFACE})
#############################################


#############################################
set( CORE_LIBRARY_SOURCES 
	${HEADER_CORE_ROOT}
    ${HEADER_CORE_DATA}
    ${HEADER_CORE_ENGINE}
	${HEADER_CORE_INTERFACE}
	${HEADER_CORE_UTILITY}
	${SOURCE_CORE_ROOT}
	${SOURCE_CORE_DATA}
	${SOURCE_CORE_ENGINE}
	${SOURCE_CORE_INTERFACE}
	${SOURCE_CORE_UTILITY} )

if( NOT CORE_USE_CUDA )
	######### Tell CMake to create the object files
	add_library( coreobj OBJECT ${CORE_LIBRARY_SOURCES} )
	set_property(TARGET coreobj PROPERTY CXX_STANDARD 11)
	set_property(TARGET coreobj PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET coreobj PROPERTY CXX_EXTENSIONS OFF)
	######### Tell CMake to create the static core library for the QT UI
	if( BUILD_UI_CXX OR BUILD_UI_WEB)
		MESSAGE( STATUS ">> Building static cxx library" )
		add_library( core STATIC $<TARGET_OBJECTS:coreobj> )
    	target_include_directories( core PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)
		target_link_libraries( core ${qhull_LIBS} )
	endif()
else()
	include_directories( core PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
	if( BUILD_UI_CXX )
		cuda_add_library( core STATIC ${CORE_LIBRARY_SOURCES} )
		target_link_libraries( core ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
endif()
#############################################

######### Tell CMake to create the shared library for the Python UI
if( BUILD_UI_PYTHON )
	set( PYLIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-python/core/ )
	file( WRITE ${PYLIB_OUTPUT_DIR}/scalar.py "import ctypes\nscalar=ctypes.c_${CORE_SCALAR_TYPE}")
	if( NOT CORE_USE_CUDA )
		MESSAGE( STATUS ">> Building shared library for Python" )
		#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
		add_library( pycore SHARED $<TARGET_OBJECTS:coreobj> )
		target_link_libraries( pycore ${qhull_LIBS} )
	else()
		MESSAGE( STATUS ">> Building shared CUDA library for Python" )
		include_directories( core PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
		cuda_add_library( pycore SHARED ${CORE_LIBRARY_SOURCES} )
		target_link_libraries( pycore ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
	### We want it to be called core, not pycore
	# set_target_properties( pycore PROPERTIES OUTPUT_NAME "core" )
	### We want it to be placed under ui-python/core/ s.t. it is directly part of the python core module/package
	set_target_properties( pycore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR} )
	if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
			set_target_properties( pycore PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${PYLIB_OUTPUT_DIR} )
		endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
	endif ( MSVC )
endif()
#############################################

######### Tell CMake to create the shared library for the Julia UI
if( BUILD_UI_JULIA )
	MESSAGE( STATUS ">> Building shared library for Julia" )
	#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
	add_library( corejl SHARED $<TARGET_OBJECTS:coreobj> )
	target_link_libraries( corejl ${qhull_LIBS})
	### We want it to be called core, not corejl
	# set_target_properties( corejl PROPERTIES OUTPUT_NAME "core" )
	### We want it to be placed under ui-julia/core/ s.t. it is directly part of the julia core module/package
	set( JLLIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-julia/core/ )
    set_target_properties( corejl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${JLLIB_OUTPUT_DIR} )
    if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        	set_target_properties( corejl PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${JLLIB_OUTPUT_DIR} )
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
    target_link_libraries( ${testName} core )
    # Properties
    set_property(TARGET ${testName} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${testName} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${testName} PROPERTY CXX_EXTENSIONS OFF)
    # Include Directories
    target_include_directories( ${testName} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMAND           ${testName} --use-colour=yes )
endmacro( add_framework_test testName testSrc )
### Create tests if needed
if ( CORE_BUILD_TEST )
	MESSAGE( STATUS ">> Building unit tests for core" )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    add_framework_test( coretest_vmath test/test_vectormath.cpp )
    add_framework_test( coretest_mmath test/test_manifoldmath.cpp )
    add_framework_test( coretest_api   test/test_api.cpp )
endif()
#############################################


######### Python Test #######################
macro(add_python_test test_name src)
	add_test(
		NAME                ${test_name}
		WORKING_DIRECTORY   ${PROJECT_BINARY_DIR}
		COMMAND             ${PYTHON_EXECUTABLE} "${PYLIB_OUTPUT_DIR}/test/${src}")

	set_tests_properties(${test_name}
		PROPERTIES ENVIRONMENT "PYTHONPATH=${PYLIB_OUTPUT_DIR}:$PYTHONPATH")
endmacro(add_python_test)
if( BUILD_UI_PYTHON )
	MESSAGE( STATUS ">> Setting up unit tests for python core" )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    # add_python_test( pycoretest_configuration   python/test/configuration.py )
endif( )
#############################################


######### Export
set(CORE_LIBRARIES        core 				       PARENT_SCOPE)
set(CORE_INCLUDE_DIRS     ${CORE_INCLUDE_DIRS}     PARENT_SCOPE)
set(HEADER_CORE_DATA      ${HEADER_CORE_DATA}      PARENT_SCOPE)
set(HEADER_CORE_ENGINE    ${HEADER_CORE_ENGINE}    PARENT_SCOPE)
set(HEADER_CORE_INTERFACE ${HEADER_CORE_INTERFACE} PARENT_SCOPE)
set(HEADER_CORE_UTILITY   ${HEADER_CORE_UTILITY}   PARENT_SCOPE)
set(SOURCE_CORE_DATA      ${SOURCE_CORE_DATA}      PARENT_SCOPE)
set(SOURCE_CORE_ENGINE    ${SOURCE_CORE_ENGINE}    PARENT_SCOPE)
set(SOURCE_CORE_INTERFACE ${SOURCE_CORE_INTERFACE} PARENT_SCOPE)
set(SOURCE_CORE_UTILITY   ${SOURCE_CORE_UTILITY}   PARENT_SCOPE)
#############################################



######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    ${HEADER_CORE_ROOT} ${HEADER_CORE_DATA} ${HEADER_CORE_ENGINE} ${HEADER_CORE_INTERFACE} ${HEADER_CORE_UTILITY}" )
	MESSAGE( STATUS ">> Sources:                    ${SOURCE_CORE_ROOT} ${SOURCE_CORE_DATA} ${SOURCE_CORE_ENGINE} ${SOURCE_CORE_INTERFACE} ${SOURCE_CORE_UTILITY}" )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- Core done ------------------------------------ <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )