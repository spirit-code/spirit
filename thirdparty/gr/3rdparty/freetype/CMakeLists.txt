######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(FREETYPE)
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
set(FREETYPE_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/freetype)
include_directories(${FREETYPE_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_FREETYPE_ROOT)
file(GLOB_RECURSE HEADER_FREETYPE_ROOT *.h)
### Source Groups
set(SOURCE_FREETYPE_ROOT src/base/ftbase.c src/base/ftbbox.c src/base/ftbitmap.c src/base/ftdebug.c src/base/ftgasp.c src/base/ftglyph.c src/base/ftinit.c src/base/ftstroke.c src/base/ftsynth.c src/base/ftsystem.c src/base/fttype1.c src/base/ftxf86.c src/cff/cff.c src/psaux/psaux.c src/pshinter/pshinter.c src/psnames/psnames.c src/raster/raster.c src/sfnt/sfnt.c src/smooth/smooth.c src/truetype/truetype.c src/type1/type1.c src/cid/type1cid.c)
# file(GLOB_RECURSE SOURCE_FREETYPE_ROOT src/base/*.c src/cff/*.c src/psaux/*.c src/pshinter/*.c src/psnames/*.c src/raster/*.c src/sfnt/*.c src/smooth/*.c src/truetype/*.c src/type1/*.c src/cid/*.c)
#############################################


######### IDE Folders #######################
#############################################


######### CFLAGS  ###########################
set(CMAKE_C_FLAGS "-DFT2_BUILD_LIBRARY -DDARWIN_NO_CARBON -DFT_CONFIG_MODULES_H=\\\"slimftmodules.h\\\" -DFT_CONFIG_OPTIONS_H=\\\"slimftoptions.h\\\"")
#############################################

######### Add Subdirectories
#############################################

######### Tell CMake to create the executable
add_library( freetype STATIC
    ${HEADER_FREETYPE_ROOT}
    ${SOURCE_FREETYPE_ROOT})
#############################################


######### Export
set(FREETYPE_LIBRARIES freetype PARENT_SCOPE)
set(FREETYPE_INCLUDE_DIRS   ${FREETYPE_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_FREETYPE_ROOT    ${HEADER_FREETYPE_ROOT} PARENT_SCOPE)
set(SOURCE_FREETYPE_ROOT    ${SOURCE_FREETYPE_ROOT} PARENT_SCOPE)

if ( PRINT_SOURCES )
    MESSAGE( STATUS ">> HEADERS:           " ${HEADER_FREETYPE_ROOT} )
    MESSAGE( STATUS ">> SOURCES:           " ${SOURCE_FREETYPE_ROOT} )
endif()