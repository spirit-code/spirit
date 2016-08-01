MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- GL ------------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(MonoSpinGL)
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
set(GL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/shaders)
include_directories(
	${GL_INCLUDE_DIRS}
	${CORE_INCLUDE_DIRS}
	${GLAD_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_GL_ROOT)
### Source Groups
set(SOURCE_GL_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/glad/src/glad.c)
#############################################

######### Add Subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/include)
#############################################

######### IDE Folders #######################
### Folder include
source_group("include" FILES ${HEADER_GL_ROOT})
### Folder src
source_group("src" FILES  ${SOURCE_GL_ROOT})
### Folder shaders
source_group("shaders" FILES
	${PROJECT_SOURCE_DIR}/shaders/arrows.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/arrows.frag.glsl
	${PROJECT_SOURCE_DIR}/shaders/hsv.colormap.glsl
	${PROJECT_SOURCE_DIR}/shaders/redblue.colormap.glsl
	${PROJECT_SOURCE_DIR}/shaders/surface.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/surface.frag.glsl
	${PROJECT_SOURCE_DIR}/shaders/sphere_points.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/sphere_points.frag.glsl
	${PROJECT_SOURCE_DIR}/shaders/sphere_background.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/sphere_background.frag.glsl
	${PROJECT_SOURCE_DIR}/shaders/boundingbox.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/boundingbox.frag.glsl
	${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.vert.glsl
	${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.frag.glsl)
#############################################

######### Tell CMake to create the executable
add_library( gl STATIC
    ${HEADER_GL_ROOT}
    ${SOURCE_GL_ROOT})
#############################################

######### Create converted shader .txt files
### Convert .glsl files to .txt files inside R"()"
### With this conversion we can read the .txt files into the code statically
add_custom_target (shaders
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/arrows.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/arrows.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/arrows.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/arrows.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/colormap.hsv.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/colormap.hsv.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/colormap.redblue.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/colormap.redblue.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/surface.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/surface.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/surface.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/surface.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/sphere_points.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/sphere_points.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/sphere_points.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/sphere_points.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/sphere_background.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/sphere_background.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/sphere_background.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/sphere_background.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/boundingbox.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/boundingbox.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/boundingbox.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/boundingbox.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.vert.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.vert.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    ${CMAKE_COMMAND} -D CFG=${PROJECT_SOURCE_DIR}/shaders/shader.in
                     -D SRC=${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.frag.glsl
                     -D DST=${PROJECT_SOURCE_DIR}/shaders/coordinatesystem.frag.txt
                     -P ${PROJECT_SOURCE_DIR}/shaders/shader.cmake
    WORKING_DIR ${PROJECT_SOURCE_DIR})
add_dependencies(gl shaders)
#############################################

######### Export
set(GL_LIBRARIES    gl PARENT_SCOPE)
set(GL_INCLUDE_DIRS ${GL_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_GL_ROOT  ${HEADER_GL_ROOT} PARENT_SCOPE)
set(SOURCE_GL_ROOT  ${SOURCE_GL_ROOT} PARENT_SCOPE)
#############################################


######### Header and Source messages ########
if ( PRINT_SOURCES )
    MESSAGE( STATUS ">> Headers:                    " ${HEADER_GL_ROOT} )
    MESSAGE( STATUS ">> Sources:                    " ${SOURCE_GL_ROOT} )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- GL done -------------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
