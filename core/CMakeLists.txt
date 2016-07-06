######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(MonoSpinCore)
#############################################

######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Where to search for library headers
set(CORE_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/data
	${PROJECT_SOURCE_DIR}/include/engine
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
### Folder src
source_group("src" FILES  ${SOURCE_CORE_ROOT}) #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
source_group("src\\data" FILES ${SOURCE_CORE_DATA})
source_group("src\\engine" FILES ${SOURCE_CORE_ENGINE})
source_group("src\\utility" FILES ${SOURCE_CORE_UTILITY})
#############################################


######### Tell CMake to create the executable
add_library( core STATIC
    ${HEADER_CORE_DATA}
    ${HEADER_CORE_ENGINE}
	${HEADER_CORE_UTILITY}
	${SOURCE_CORE_DATA}
	${SOURCE_CORE_ENGINE}
	${SOURCE_CORE_UTILITY})
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
set(HEADER_CORE_UTILITY ${HEADER_CORE_UTILITY} PARENT_SCOPE)
set(SOURCE_CORE_DATA    ${SOURCE_CORE_DATA} PARENT_SCOPE)
set(SOURCE_CORE_ENGINE  ${SOURCE_CORE_ENGINE} PARENT_SCOPE)
set(SOURCE_CORE_UTILITY ${SOURCE_CORE_UTILITY} PARENT_SCOPE)