MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Spirit --------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
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
set(META_VERSION_MINOR        "3")
set(META_VERSION_PATCH        "0")
set(META_VERSION_REVISION     "${GIT_REV}")
set(META_VERSION              "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}")
set(META_NAME_VERSION         "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})")
#############################################

######### Project Name ######################
project(${META_PROJECT_NAME} VERSION ${META_VERSION})
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


######### Generate Spirit_Defines.h ################
set(VERSION_STRING "${META_VERSION}")
set(VERSION_REVISION_STRING "${META_VERSION_REVISION}")
string(TOUPPER ${SPIRIT_SCALAR_TYPE} SPIRIT_SCALAR_TYPE_UPPERCASE)
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
	 ${PROJECT_SOURCE_DIR}/include/interface
	 ${PROJECT_SOURCE_DIR}/include/utility 
	 ${PROJECT_SOURCE_DIR}/src/utility )
include_directories( ${SPIRIT_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${SPECTRA_INCLUDE_DIRS} ${QHULL_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_SPIRIT_ROOT)
set(HEADER_SPIRIT_ENGINE)
set(HEADER_SPIRIT_UTILITY)
### Source Groups
set(SOURCE_SPIRIT_ROOT)
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
source_group("include\\interface" FILES ${HEADER_SPIRIT_INTERFACE})
### Folder src
source_group("src" FILES  ${SOURCE_SPIRIT_ROOT}) #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
source_group("src\\data" FILES ${SOURCE_SPIRIT_DATA})
source_group("src\\engine" FILES ${SOURCE_SPIRIT_ENGINE})
source_group("src\\utility" FILES ${SOURCE_SPIRIT_UTILITY})
source_group("src\\interface" FILES ${SOURCE_SPIRIT_INTERFACE})
#############################################


#############################################
set( SPIRIT_LIBRARY_SOURCES 
	${HEADER_SPIRIT_ROOT}
    ${HEADER_SPIRIT_DATA}
    ${HEADER_SPIRIT_ENGINE}
	${HEADER_SPIRIT_INTERFACE}
	${HEADER_SPIRIT_UTILITY}
	${SOURCE_SPIRIT_ROOT}
	${SOURCE_SPIRIT_DATA}
	${SOURCE_SPIRIT_ENGINE}
	${SOURCE_SPIRIT_INTERFACE}
	${SOURCE_SPIRIT_UTILITY} )

if( NOT SPIRIT_USE_CUDA )
	######### Tell CMake to create the object files
	add_library( spirit OBJECT ${SPIRIT_LIBRARY_SOURCES} )
	set_property(TARGET spirit PROPERTY CXX_STANDARD 11)
	set_property(TARGET spirit PROPERTY CXX_STANDARD_REQUIRED ON)
	set_property(TARGET spirit PROPERTY CXX_EXTENSIONS OFF)
	######### Tell CMake to create the static spirit library for the QT UI
	if( BUILD_UI_CXX OR BUILD_UI_WEB)
		MESSAGE( STATUS ">> Building static cxx library" )
		add_library( spirit_static STATIC $<TARGET_OBJECTS:spirit> )
    	target_include_directories( spirit_static PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)
		target_link_libraries( spirit_static ${qhull_LIBS} )
		set_target_properties( spirit_static PROPERTIES OUTPUT_NAME "spirit" )
	endif()
else()
	include_directories( spirit_static PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
	if( BUILD_UI_CXX )
		cuda_add_library( spirit_static STATIC ${SPIRIT_LIBRARY_SOURCES} )
		target_link_libraries( spirit_static ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
endif()
#############################################

######### Tell CMake to create the shared library for the Python UI
if( BUILD_UI_PYTHON )
	set( PYLIB_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/python/Spirit/ )
	file( WRITE ${PYLIB_OUTPUT_DIR}/scalar.py "import ctypes\nscalar=ctypes.c_${SPIRIT_SCALAR_TYPE}")
	if( NOT SPIRIT_USE_CUDA )
		MESSAGE( STATUS ">> Building shared library for Python" )
		#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
		add_library( spirit_python SHARED $<TARGET_OBJECTS:spirit> )
		target_link_libraries( spirit_python ${qhull_LIBS} )
	else()
		MESSAGE( STATUS ">> Building shared CUDA library for Python" )
		include_directories( spirit_python PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
		cuda_add_library( spirit_python SHARED ${SPIRIT_LIBRARY_SOURCES} )
		target_link_libraries( spirit_python ${qhull_LIBS} ${CUDA_LIBRARIES} )
	endif()
	### We want it to be called spirit, not spirit_python
	set_target_properties( spirit_python PROPERTIES OUTPUT_NAME "spirit" )
	### We want it to be placed under python/Spirit/ s.t. it is directly part of the python Spirit bindings module/package
	set_target_properties( spirit_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR} )
	if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
			set_target_properties( spirit_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${PYLIB_OUTPUT_DIR} )
		endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
	endif ( MSVC )
endif()
#############################################

######### Tell CMake to create the shared library for the Julia UI
if( BUILD_UI_JULIA )
	MESSAGE( STATUS ">> Building shared library for Julia" )
	#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
	add_library( spirit_julia SHARED $<TARGET_OBJECTS:spirit> )
	target_link_libraries( spirit_julia ${qhull_LIBS})
	### We want it to be called spirit, not spirit_julia
	set_target_properties( spirit_julia PROPERTIES OUTPUT_NAME "spirit" )
	### We want it to be placed under julia/Spirit/ s.t. it is directly part of the julia spirit bindings module/package
	set( JLLIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-julia/spirit/ )
    set_target_properties( spirit_julia PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${JLLIB_OUTPUT_DIR} )
    if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        	set_target_properties( spirit_julia PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${JLLIB_OUTPUT_DIR} )
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
    target_link_libraries( ${testName} spirit_static )
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
if ( SPIRIT_BUILD_TEST )
	MESSAGE( STATUS ">> Building unit tests for Spirit" )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    add_framework_test( test_vmath test/test_vectormath.cpp )
    add_framework_test( test_mmath test/test_manifoldmath.cpp )
    add_framework_test( test_api   test/test_api.cpp )
endif()
#############################################


######### Python Test #######################
macro(add_python_test test_name src)
	# Add the test
	add_test(
		NAME                ${test_name}
		WORKING_DIRECTORY   ${PROJECT_BINARY_DIR}
		COMMAND             ${PYTHON_EXECUTABLE} "${PYLIB_OUTPUT_DIR}/test/${src}")
	# Properties
	set_tests_properties(${test_name}
		PROPERTIES ENVIRONMENT "PYTHONPATH=${PYLIB_OUTPUT_DIR}:$PYTHONPATH")
endmacro(add_python_test)
### Create tests if needed
if( BUILD_UI_PYTHON )
	MESSAGE( STATUS ">> Setting up unit tests for python bindings" )

    ### Enable CTest testing
    enable_testing()

    ### Tests
    # add_python_test( test_python_configuration   python/test/configuration.py )
endif( )
#############################################


######### Export
set(SPIRIT_LIBRARIES        spirit			       	   PARENT_SCOPE)
set(SPIRIT_LIBRARIES_STATIC spirit_static		       PARENT_SCOPE)
set(SPIRIT_INCLUDE_DIRS     ${SPIRIT_INCLUDE_DIRS}     PARENT_SCOPE)
set(HEADER_SPIRIT_DATA      ${HEADER_SPIRIT_DATA}      PARENT_SCOPE)
set(HEADER_SPIRIT_ENGINE    ${HEADER_SPIRIT_ENGINE}    PARENT_SCOPE)
set(HEADER_SPIRIT_INTERFACE ${HEADER_SPIRIT_INTERFACE} PARENT_SCOPE)
set(HEADER_SPIRIT_UTILITY   ${HEADER_SPIRIT_UTILITY}   PARENT_SCOPE)
set(SOURCE_SPIRIT_DATA      ${SOURCE_SPIRIT_DATA}      PARENT_SCOPE)
set(SOURCE_SPIRIT_ENGINE    ${SOURCE_SPIRIT_ENGINE}    PARENT_SCOPE)
set(SOURCE_SPIRIT_INTERFACE ${SOURCE_SPIRIT_INTERFACE} PARENT_SCOPE)
set(SOURCE_SPIRIT_UTILITY   ${SOURCE_SPIRIT_UTILITY}   PARENT_SCOPE)
#############################################



######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    ${HEADER_SPIRIT_ROOT} ${HEADER_SPIRIT_DATA} ${HEADER_SPIRIT_ENGINE} ${HEADER_SPIRIT_INTERFACE} ${HEADER_SPIRIT_UTILITY}" )
	MESSAGE( STATUS ">> Sources:                    ${SOURCE_SPIRIT_ROOT} ${SOURCE_SPIRIT_DATA} ${SOURCE_SPIRIT_ENGINE} ${SOURCE_SPIRIT_INTERFACE} ${SOURCE_SPIRIT_UTILITY}" )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- Spirit done ---------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )