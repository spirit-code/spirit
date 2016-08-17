######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(GR)
#############################################

######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Where to search for library headers
set(GR_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR})
include_directories(${GR_INCLUDE_DIRS}, ${GKS_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_GR_ROOT)
file(GLOB HEADER_GR_ROOT *.h)
### Source Groups
set(SOURCE_GR_ROOT)
file(GLOB SOURCE_GR_ROOT *.c)
#############################################


######### IDE Folders #######################
#############################################


######### CFLAGS  ###########################
#############################################

######### Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/qtgr)
#############################################

######### Tell CMake to create the executable
add_library( gr STATIC
    ${HEADER_GR_ROOT}
    ${SOURCE_GR_ROOT})
target_link_libraries(gr ${GKS_LIBRARIES})
#############################################


######### Export
set(GR_LIBRARIES gr PARENT_SCOPE)
set(GR_INCLUDE_DIRS   ${GR_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_GR_ROOT    ${HEADER_GR_ROOT} PARENT_SCOPE)
set(SOURCE_GR_ROOT    ${SOURCE_GR_ROOT} PARENT_SCOPE)
set(QTGR_LIBRARIES qtgr PARENT_SCOPE)
set(QTGR_INCLUDE_DIRS   ${QTGR_INCLUDE_DIRS} PARENT_SCOPE)

if ( PRINT_SOURCES )
    MESSAGE( STATUS ">> HEADERS:           " ${HEADER_GR_ROOT} )
    MESSAGE( STATUS ">> SOURCES:           " ${SOURCE_GR_ROOT} )
endif()