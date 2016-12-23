cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(VFRendering VERSION 0.3.0)

option(BUILD_DEMO "Whether or not a demo executable should be built" OFF)

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


set(SOURCE_FILES
    src/ArrowRenderer.cxx
    src/BoundingBoxRenderer.cxx
    src/CombinedRenderer.cxx
    src/CoordinateSystemRenderer.cxx
    src/FPSCounter.cxx
    src/Geometry.cxx
    src/IsosurfaceRenderer.cxx
    src/Options.cxx
    src/RendererBase.cxx
    src/SurfaceRenderer.cxx
    src/Utilities.cxx
    src/VectorfieldIsosurface.cxx
    src/VectorSphereRenderer.cxx
    src/View.cxx
    ${PROJECT_SOURCE_DIR}/thirdparty/glad/src/glad.c
)

set(HEADER_FILES
    include/VectorfieldIsosurface.hxx
    include/VFRendering/ArrowRenderer.hxx
    include/VFRendering/BoundingBoxRenderer.hxx
    include/VFRendering/CombinedRenderer.hxx
    include/VFRendering/CoordinateSystemRenderer.hxx
    include/VFRendering/FPSCounter.hxx
    include/VFRendering/Geometry.hxx
    include/VFRendering/IsosurfaceRenderer.hxx
    include/VFRendering/Options.hxx
    include/VFRendering/RendererBase.hxx
    include/VFRendering/SurfaceRenderer.hxx
    include/VFRendering/Utilities.hxx
    include/VFRendering/VectorSphereRenderer.hxx
    include/VFRendering/View.hxx
    include/shaders
    include/shaders/arrows.frag.glsl.hxx
    include/shaders/arrows.vert.glsl.hxx
    include/shaders/boundingbox.frag.glsl.hxx
    include/shaders/boundingbox.vert.glsl.hxx
    include/shaders/colormap.bluegreenred.glsl.hxx
    include/shaders/colormap.bluered.glsl.hxx
    include/shaders/colormap.bluewhitered.glsl.hxx
    include/shaders/colormap.hsv.glsl.hxx
    include/shaders/coordinatesystem.frag.glsl.hxx
    include/shaders/coordinatesystem.vert.glsl.hxx
    include/shaders/sphere_background.frag.glsl.hxx
    include/shaders/sphere_background.vert.glsl.hxx
    include/shaders/sphere_points.frag.glsl.hxx
    include/shaders/sphere_points.vert.glsl.hxx
    include/shaders/surface.frag.glsl.hxx
    include/shaders/surface.vert.glsl.hxx
)

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

find_package(glfw3 3.2 REQUIRED)
target_link_libraries(${PROJECT_NAME}Demo glfw)

set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME}Demo PROPERTY CXX_EXTENSIONS OFF)

target_include_directories(${PROJECT_NAME}Demo PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/glad/include)
endif()
