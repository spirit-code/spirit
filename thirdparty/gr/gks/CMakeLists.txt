######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(GKS)
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
set(GKS_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR})
include_directories(${GKS_INCLUDE_DIRS},
                    ${CMAKE_SOURCE_DIR}/thirdparty/gr/3rdparty/freetype/include/freetype)
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_GKS_ROOT)
file(GLOB HEADER_GKS_ROOT *.h)
### Source Groups
set(SOURCE_GKS_ROOT)
file(GLOB SOURCE_GKS_ROOT *.c)
#############################################

######### IDE Folders #######################
#############################################


######### CFLAGS  ###########################
#############################################


######### Tell CMake to create the executable
add_library( gks STATIC
    ${HEADER_GKS_ROOT}
    ${SOURCE_GKS_ROOT})
#############################################

set(GKS_LIBRARIES gks)
set(GKS_INCLUDE_DIRS   ${GKS_INCLUDE_DIRS})
######### Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/plugin)
#############################################

######### copy font files ###################
file(GLOB GKS_FONT_FILES ${PROJECT_SOURCE_DIR}/fonts/*)
file(COPY ${GKS_FONT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/fonts/)

######### Export
set(GKS_LIBRARIES gks PARENT_SCOPE)
set(GKS_INCLUDE_DIRS   ${GKS_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_GKS_ROOT    ${HEADER_GKS_ROOT} PARENT_SCOPE)
set(SOURCE_GKS_ROOT    ${SOURCE_GKS_ROOT} PARENT_SCOPE)

MESSAGE( STATUS ">> HEADERS:           " ${HEADER_GKS_ROOT} )
MESSAGE( STATUS ">> SOURCES:           " ${SOURCE_GKS_ROOT} )


