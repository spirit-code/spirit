MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Core ----------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(SpiritCore)
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
# set(CMAKE_DISABLE_SOURCE_CHANGES ON) # we need source changes for the generated version.h
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Generate Version.h ################
set(VERSION_STRING "${META_VERSION}")
set(VERSION_REVISION_STRING "${META_VERSION_REVISION}")
configure_file(${CMAKE_SOURCE_DIR}/CMake/Version.h.in ${PROJECT_SOURCE_DIR}/include/Version.h)
#############################################


######### Where to search for library headers
set(CORE_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/data
	${PROJECT_SOURCE_DIR}/include/engine
	${PROJECT_SOURCE_DIR}/include/interface
	${PROJECT_SOURCE_DIR}/include/utility
	${PROJECT_SOURCE_DIR}/src/utility)
include_directories(${CORE_INCLUDE_DIRS})
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


######### Tell CMake to create the static library
add_library( core STATIC
	${HEADER_CORE_ROOT}
    ${HEADER_CORE_DATA}
    ${HEADER_CORE_ENGINE}
	${HEADER_CORE_INTERFACE}
	${HEADER_CORE_UTILITY}
	${SOURCE_CORE_ROOT}
	${SOURCE_CORE_DATA}
	${SOURCE_CORE_ENGINE}
	${SOURCE_CORE_INTERFACE}
	${SOURCE_CORE_UTILITY})
#############################################

######### Tell CMake to create the shared library
if( BUILD_UI_PYTHON )
	MESSAGE( STATUS ">> Building shared library for Python" )
	### We dont use dll_export directives, so on Windows we just export everything
	set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON )
	#SET( CMAKE_SHARED_LIBRARY_SUFFIX ".so" )
	add_library( pycore SHARED
		${HEADER_CORE_ROOT}
		${HEADER_CORE_DATA}
		${HEADER_CORE_ENGINE}
		${HEADER_CORE_INTERFACE}
		${HEADER_CORE_UTILITY}
		${SOURCE_CORE_ROOT}
		${SOURCE_CORE_DATA}
		${SOURCE_CORE_ENGINE}
		${SOURCE_CORE_INTERFACE}
		${SOURCE_CORE_UTILITY})
	### We want it to be called core, not pycore
	set_target_properties( pycore PROPERTIES OUTPUT_NAME "core" )
	### We want it to be placed under ui-python/core/ s.t. it is directly part of the python core module/package
	set( PYLIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-python/core/ )
    set_target_properties( pycore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_OUTPUT_DIR} )
    if ( MSVC )
		foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
			string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
        	set_target_properties( pycore PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG}          ${PYLIB_OUTPUT_DIR} )
		endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
    endif ( MSVC )
endif()
#############################################


### TODO: Add threading define if we dont use emscripten (which cannot use threads)
# if (NOT BUILD_UI_WEB)
# 	target_compile_definitions( core "USE_THREADS")
# endif()
### or use add_definitions() to add the define to all targets


######### Export
set(CORE_LIBRARIES core PARENT_SCOPE)
set(CORE_INCLUDE_DIRS   ${CORE_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_CORE_DATA    ${HEADER_CORE_DATA} PARENT_SCOPE)
set(HEADER_CORE_ENGINE  ${HEADER_CORE_ENGINE} PARENT_SCOPE)
set(HEADER_CORE_INTERFACE ${HEADER_CORE_INTERFACE} PARENT_SCOPE)
set(HEADER_CORE_UTILITY ${HEADER_CORE_UTILITY} PARENT_SCOPE)
set(SOURCE_CORE_DATA    ${SOURCE_CORE_DATA} PARENT_SCOPE)
set(SOURCE_CORE_ENGINE  ${SOURCE_CORE_ENGINE} PARENT_SCOPE)
set(SOURCE_CORE_INTERFACE ${SOURCE_CORE_INTERFACE} PARENT_SCOPE)
set(SOURCE_CORE_UTILITY ${SOURCE_CORE_UTILITY} PARENT_SCOPE)
#############################################




######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    ${HEADER_CORE_ROOT} ${HEADER_CORE_DATA} ${HEADER_CORE_ENGINE} ${HEADER_CORE_INTERFACE} ${HEADER_CORE_UTILITY}" )
	MESSAGE( STATUS ">> Sources:                    ${SOURCE_CORE_ROOT} ${SOURCE_CORE_DATA} ${SOURCE_CORE_ENGINE} ${SOURCE_CORE_INTERFACE} ${SOURCE_CORE_UTILITY}" )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- Core done ------------------------------------ <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )