cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

file(STRINGS "Version.txt" VFRENDERING_VERSION)
project(VFRendering VERSION ${VFRENDERING_VERSION})

option(BUILD_DEMO "Whether or not a demo executable should be built" OFF)
option(BUILD_PYTHON_BINDINGS "Whether or not a binary python module should be built" OFF)

if((NOT qhull_LIBS) OR (NOT qhull_INCLUDE_DIRS))

if(qhull_LIBS)
message(WARNING "qhull_LIBS is set, but qhull_INCLUDE_DIRS is missing.")
endif()
if(qhull_INCLUDE_DIRS)
message(WARNING "qhull_INCLUDE_DIRS is set, but qhull_LIBS is missing.")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ExternalProject)
ExternalProject_add(qhull
    GIT_REPOSITORY https://github.com/qhull/qhull.git
    GIT_TAG v7.3.1
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/thirdparty-install;-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    PREFIX qhull-prefix
)

### qhullstatic reentrant library
add_library(libqhullstatic_r STATIC IMPORTED)
set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
if ( WIN32 )
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER )
        set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_${OUTPUTCONFIG_UPPER} ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${OUTPUTCONFIG}/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
endif ( WIN32 )
add_dependencies(libqhullstatic_r qhull)

### qhullcpp library
add_library(libqhullcpp STATIC IMPORTED)
set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
if ( WIN32 )
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER )
        set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_${OUTPUTCONFIG_UPPER} ${PROJECT_BINARY_DIR}/qhull-prefix/src/qhull-build/${OUTPUTCONFIG}/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
endif ( WIN32 )
set_property(TARGET libqhullcpp PROPERTY INTERFACE_LINK_LIBRARIES libqhullstatic_r)
add_dependencies(libqhullcpp qhull)

set(qhull_LIBS libqhullcpp)
### Add corresponding include directories
set(qhull_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/thirdparty-install/include;${PROJECT_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
endif()


set(SOURCE_FILES
    src/ArrowRenderer.cxx
    src/BoundingBoxRenderer.cxx
    src/ParallelepipedRenderer.cxx 
    src/CombinedRenderer.cxx
    src/CoordinateSystemRenderer.cxx
    src/FPSCounter.cxx
    src/Geometry.cxx
    src/GlyphRenderer.cxx
    src/DotRenderer.cxx
    src/IsosurfaceRenderer.cxx
    src/Options.cxx
    src/RendererBase.cxx
    src/SphereRenderer.cxx
    src/SurfaceRenderer.cxx
    src/Utilities.cxx
    src/VectorField.cxx
    src/VectorFieldRenderer.cxx
    src/VectorfieldIsosurface.cxx
    src/VectorSphereRenderer.cxx
    src/View.cxx
    ${PROJECT_SOURCE_DIR}/thirdparty/glad/src/glad.c
)

set(HEADER_FILES
    include/VectorfieldIsosurface.hxx
    include/VFRendering/ArrowRenderer.hxx
    include/VFRendering/BoundingBoxRenderer.hxx
    include/VFRendering/ParallelepipedRenderer.hxx
    include/VFRendering/CombinedRenderer.hxx
    include/VFRendering/CoordinateSystemRenderer.hxx
    include/VFRendering/FPSCounter.hxx
    include/VFRendering/GlyphRenderer.hxx
    include/VFRendering/DotRenderer.hxx
    include/VFRendering/Geometry.hxx
    include/VFRendering/IsosurfaceRenderer.hxx
    include/VFRendering/Options.hxx
    include/VFRendering/RendererBase.hxx
    include/VFRendering/SphereRenderer.hxx
    include/VFRendering/SurfaceRenderer.hxx
    include/VFRendering/Utilities.hxx
    include/VFRendering/VectorField.hxx
    include/VFRendering/VectorFieldRenderer.hxx
    include/VFRendering/VectorSphereRenderer.hxx
    include/VFRendering/View.hxx
    include/shaders
    include/shaders/dot.frag.glsl.hxx
    include/shaders/dot.vert.glsl.hxx
    include/shaders/glyphs.frag.glsl.hxx
    include/shaders/glyphs.vert.glsl.hxx
    include/shaders/boundingbox.frag.glsl.hxx
    include/shaders/boundingbox.vert.glsl.hxx
    include/shaders/colormap.bluegreenred.glsl.hxx
    include/shaders/colormap.bluered.glsl.hxx
    include/shaders/colormap.bluewhitered.glsl.hxx
    include/shaders/colormap.hsv.glsl.hxx
    include/shaders/colormap.black.glsl.hxx
    include/shaders/colormap.white.glsl.hxx
    include/shaders/coordinatesystem.frag.glsl.hxx
    include/shaders/coordinatesystem.vert.glsl.hxx
    include/shaders/sphere_background.frag.glsl.hxx
    include/shaders/sphere_background.vert.glsl.hxx
    include/shaders/sphere_points.frag.glsl.hxx
    include/shaders/sphere_points.vert.glsl.hxx
    include/shaders/surface.frag.glsl.hxx
    include/shaders/surface.vert.glsl.hxx
    include/shaders/isosurface.frag.glsl.hxx
    include/shaders/isosurface.vert.glsl.hxx
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

add_library(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${qhull_LIBS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty/glm/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/glad/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${qhull_INCLUDE_DIRS})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/VFRendering DESTINATION include)

# Build demo
if (BUILD_DEMO)
add_executable(${PROJECT_NAME}Demo demo.cxx)
target_link_libraries(${PROJECT_NAME}Demo ${PROJECT_NAME})
if (${UNIX})
target_link_libraries(${PROJECT_NAME}Demo dl)
endif(${UNIX})

find_package(glfw3 3 REQUIRED)
target_link_libraries(${PROJECT_NAME}Demo glfw)

set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${PROJECT_NAME}Demo PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/glad/include)
endif()

# Build Python bindings
if (BUILD_PYTHON_BINDINGS)
set(MODULE_NAME py${PROJECT_NAME})
add_subdirectory(thirdparty/pybind11)
pybind11_add_module(${MODULE_NAME} python/vfrendering_bindings.cpp)

target_compile_definitions(${MODULE_NAME} PUBLIC "-D${PROJECT_NAME}_VERSION=\"${${PROJECT_NAME}_VERSION}${MODULE_DEV_TAG}\"")

target_include_directories(${MODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${MODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty/pybind11/include)
target_include_directories(${MODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty/glm/include)
target_include_directories(${MODULE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/glad/include)
target_include_directories(${MODULE_NAME} PRIVATE ${qhull_INCLUDE_DIRS})

target_link_libraries(${MODULE_NAME} PRIVATE ${PROJECT_NAME})
endif()