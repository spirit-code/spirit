#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "pybind11_glm.hpp"

#include <VFRendering/View.hxx>
#include <VFRendering/VectorField.hxx>
#include <VFRendering/Geometry.hxx>
#include <VFRendering/RendererBase.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/DotRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/ParallelepipedRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/SurfaceRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/VectorSphereRenderer.hxx>
#include <VFRendering/Options.hxx>

#include <memory>

namespace py = pybind11;
using namespace VFRendering;

typedef unsigned int index_type;

PYBIND11_MODULE(pyVFRendering, m)
{
    #ifdef VFRendering_VERSION
        m.attr("__version__") = VFRendering_VERSION;
    #else
        m.attr("__version__") = "dev";
    #endif


    // Module docstring
    m.doc() = "VFRendering is a C++ library for rendering vectorfields using OpenGL,"
              "wrapped for Python using pybind11.";


    // Module functions
    m.def("getColormapImplementation", &Utilities::getColormapImplementation,
        "Get a Colormap implementation from the Colormap enum");


    // Geometry class
    py::class_<Geometry>(m, "Geometry",
        "This class defines the positions of the data points and the triangulation indices between the data points."
        " In 2D, a triangulation can be used to display data on a smooth surface of triangles."
        " In 3D, tetrahedra can be used e.g. to calculate isosurfaces.")
        .def(py::init<>())
        .def(py::init<const std::vector<glm::vec3>&, const std::vector<std::array<index_type, 3>>&, const std::vector<std::array<index_type, 4>>&, const bool&>())
        .def("cartesianGeometry", &Geometry::cartesianGeometry,
            "Create a geometry corresponding to a cartesian grid")
        .def("rectilinearGeometry", &Geometry::rectilinearGeometry,
            "Create a geometry corresponding to a rectilinear grid")
        .def("positions", &Geometry::positions,
            "Retrieve the positions given by the geometry")
        .def("min", &Geometry::min,
            "Retrieve the x-, y- and z- minimum values of all positions")
        .def("max", &Geometry::max,
            "Retrieve the x-, y- and z- maximum values of all positions")
        .def("surfaceIndices", &Geometry::surfaceIndices,
            "Retrieve the triangle indices of a 2D surface")
        .def("volumeIndices", &Geometry::volumeIndices,
            "Retrieve the tetrahedra indices of a 3D volume")
        .def("is2d", &Geometry::is2d,
            "Returns true if the geometry is planar");

    // VectorField
    py::class_<VectorField>(m, "VectorField",
        "This class holds Geometry- and Vector-data which can be used by one or multiple Renderers.")
        .def(py::init<const Geometry&,const std::vector<glm::vec3>&>())
        .def("update", &VectorField::update,
            "Update the geometry and directions of this VectorField")
        .def("updateGeometry", &VectorField::updateGeometry,
            "Update the geometry")
        .def("updateVectors", &VectorField::updateVectors,
            "Update the direction vectors")
        .def("getPositions", &VectorField::positions,
            "Get the positions stored in the VectorField")
        .def("getDirections", &VectorField::directions,
            "Get the directions stored in the VectorField")
        .def("getSurfaceIndices", &VectorField::surfaceIndices,
            "Get the surface indices stored in the VectorField")
        .def("getVolumeIndices", &VectorField::volumeIndices,
            "Get the volume indices stored in the VectorField");

    // View
    py::class_<View>(m, "View",
        "This class holds the Renderers and global Options and is used to draw into the current OpenGL context.")
        // Constructor
        .def(py::init<>())
        // Actions
        .def("draw", &View::draw,
            "Draw into current OpenGL context")
        .def("updateOptions", &View::updateOptions,
            "Update the set of options given to the View")
        .def("mouseMove", &View::mouseMove,
            "Influence the camera according to a movement of the mouse")
        .def("mouseScroll", &View::mouseScroll,
            "Influence the camera according to a scrolling of the mouse")
        // Getters
        .def("renderers", &View::renderers,
            "Retrieve the renderers currently in use by the View")
        .def("options", (const Options& (View::*)() const) &View::options,
            "Retrieve the options currently in use by the View")
        .def("getFramerate", &View::getFramerate,
            "Retrieve the last known framerate of OpenGL draws")
        // Setters
        .def("setFramebufferSize", &View::setFramebufferSize,
            "Set the size of the Framebuffer into which the View should render, i.e. the number of pixels")
        // Camera set
        .def("setVerticalFOV",             &View::setOption<View::Option::VERTICAL_FIELD_OF_VIEW>,
            "Set the vertical field of view (FOV) of the camera")
        .def("setSystemCenter",            &View::setOption<View::Option::SYSTEM_CENTER>,
            "Set the system center, i.e. the center point of the system's bounds")
        .def("setCameraPosition",          &View::setOption<View::Option::CAMERA_POSITION>,
            "Set the position of the camera")
        .def("setCenterPosition",          &View::setOption<View::Option::CENTER_POSITION>,
            "Set the center position, i.e. the focal point of the View")
        .def("setUpVector",                &View::setOption<View::Option::UP_VECTOR>,
            "Set the up-vector of the camera")
        // Colors set
        .def("setBackgroundColor",         &View::setOption<View::Option::BACKGROUND_COLOR>,
            "Set the background color of the View")
        .def("setColormapImplementation",  &View::setOption<View::Option::COLORMAP_IMPLEMENTATION>,
            "Set the implementation of the colormap")
        // Filters set
        .def("setIsVisibleImplementation", &View::setOption<View::Option::IS_VISIBLE_IMPLEMENTATION>,
            "Set a filter for the visibility of objects");


    // View Options
    py::class_<Options>(m, "Options",
        "This class holds the View's various options.")
        // Constructor
        .def(py::init<>())
        // Camera get
        .def("getVerticalFOV",             &Options::get<View::Option::VERTICAL_FIELD_OF_VIEW>,
            "Retrieve the vertical field of view (FOV) of the camera")
        .def("getSystemCenter",            &Options::get<View::Option::SYSTEM_CENTER>,
            "Retrieve the system center, i.e. the center point of the system's bounds")
        .def("getCameraPosition",          &Options::get<View::Option::CAMERA_POSITION>,
            "Retrieve the position of the camera")
        .def("getCenterPosition",          &Options::get<View::Option::CENTER_POSITION>,
            "Retrieve the center position, i.e. the focal point of the View")
        .def("getUpVector",                &Options::get<View::Option::UP_VECTOR>,
            "Retrieve the up-vector of the camera")
        // Camera set
        .def("setVerticalFOV",             &Options::set<View::Option::VERTICAL_FIELD_OF_VIEW>,
            "Set the vertical field of view (FOV) of the camera")
        .def("setSystemCenter",            &Options::set<View::Option::SYSTEM_CENTER>,
            "Set the system center, i.e. the center point of the system's bounds")
        .def("setCameraPosition",          &Options::set<View::Option::CAMERA_POSITION>,
            "Set the position of the camera")
        .def("setCenterPosition",          &Options::set<View::Option::CENTER_POSITION>,
            "Set the center position, i.e. the focal point of the View")
        .def("setUpVector",                &Options::set<View::Option::UP_VECTOR>,
            "Set the up-vector of the camera")
        // Colors get
        .def("getBackgroundColor",         &Options::get<View::Option::BACKGROUND_COLOR>,
            "Retrieve the background color of the View")
        // Colors set
        .def("setBackgroundColor",         &Options::set<View::Option::BACKGROUND_COLOR>,
            "Set the background color of the View")
        .def("setColormapImplementation",  &Options::set<View::Option::COLORMAP_IMPLEMENTATION>,
            "Set the implementation of the colormap")
        // Filters set
        .def("setIsVisibleImplementation", &Options::set<View::Option::IS_VISIBLE_IMPLEMENTATION>,
            "Set a filter for the visibility of objects");


    // Colormap enum
    py::enum_<Utilities::Colormap>(m, "Colormap")
        .value("hsv",            Utilities::Colormap::HSV)
        .value("blue_white_red", Utilities::Colormap::BLUEWHITERED)
        .value("blue_green_red", Utilities::Colormap::BLUEGREENRED)
        .value("black",          Utilities::Colormap::BLACK)
        .value("white",          Utilities::Colormap::WHITE)
        .export_values();


    // Camera movement modes enum to be used in View::mouseMove
    py::enum_<CameraMovementModes>(m, "CameraMovementModes")
        .value("rotate_bounded", CameraMovementModes::ROTATE_BOUNDED)
        .value("rotate_free",    CameraMovementModes::ROTATE_FREE)
        .value("translate",      CameraMovementModes::TRANSLATE)
        .export_values();

    
    // Styles of the dot drawn by the DotRenderer
    py::enum_<DotRenderer::DotStyle>(m, "DotRendererStyle")
        .value("circle", DotRenderer::DotStyle::CIRCLE)
        .value("square", DotRenderer::DotStyle::SQUARE)
        .export_values();


    // Renderer base class
    py::class_<RendererBase, std::shared_ptr<RendererBase>>(m, "RendererBase", "Renderer base class");


    // Combined renderer
    py::class_<CombinedRenderer, RendererBase, std::shared_ptr<CombinedRenderer>>(m, "CombinedRenderer",
        "Class for using multiple renderers in one")
        .def(py::init<const View&, const std::vector<std::shared_ptr<RendererBase>>&>());


    // ArrowRenderer
    py::class_<ArrowRenderer, RendererBase, std::shared_ptr<ArrowRenderer>>(m, "ArrowRenderer",
        "This class is used to draw arrows directly corresponding to a vectorfield.")
        .def(py::init<View&, VectorField&>())
        .def("setLevelOfDetail",  &ArrowRenderer::setOption<ArrowRenderer::Option::LEVEL_OF_DETAIL>,
            "Set the resolution of an arrow")
        .def("setConeRadius",  &ArrowRenderer::setOption<ArrowRenderer::Option::CONE_RADIUS>,
            "Set the cone radius of an arrow")
        .def("setConeHeight",  &ArrowRenderer::setOption<ArrowRenderer::Option::CONE_HEIGHT>,
            "Set the cone height of an arrow")
        .def("setCylinderRadius",  &ArrowRenderer::setOption<ArrowRenderer::Option::CYLINDER_RADIUS>,
            "Set the cylinder radius of an arrow")
        .def("setCylinderHeight",  &ArrowRenderer::setOption<ArrowRenderer::Option::CYLINDER_HEIGHT>,
            "Set the cylinder height of an arrow");

    // SphereRenderer
    py::class_<SphereRenderer, RendererBase, std::shared_ptr<SphereRenderer>>(m, "SphereRenderer",
        "This class is used to draw spheres at the positions of vectorfield, with colors corresponding to direction.")
        .def(py::init<View&, VectorField&>())
        .def("setLevelOfDetail",  &SphereRenderer::setOption<SphereRenderer::Option::LEVEL_OF_DETAIL>,
            "Set the resolution of a sphere")
        .def("setSphereRadius",  &SphereRenderer::setOption<SphereRenderer::Option::SPHERE_RADIUS>,
            "Set the radius of a sphere");

    // ParallelepipedRenderer
    py::class_<ParallelepipedRenderer, RendererBase, std::shared_ptr<ParallelepipedRenderer>>(m, "ParallelepipedRenderer",
        "This class is used to draw parallelepiped at the positions of vectorfield, with colors corresponding to direction.")
        .def(py::init<View&, VectorField&>())
        .def("setParallelepipedLengthA",  &ParallelepipedRenderer::setOption<ParallelepipedRenderer::Option::LENGTH_A>,
            "Set the length a of the parallelepiped")
        .def("setParallelepipedLengthB",  &ParallelepipedRenderer::setOption<ParallelepipedRenderer::Option::LENGTH_B>,
            "Set the length b of the parallelepiped")
        .def("setParallelepipedLengthC",  &ParallelepipedRenderer::setOption<ParallelepipedRenderer::Option::LENGTH_C>,
            "Set the length c of the parallelepiped");

    // BoundingBoxRenderer
    py::class_<BoundingBoxRenderer, RendererBase, std::shared_ptr<BoundingBoxRenderer>>(m, "BoundingBoxRenderer",
        "This Renderer draws a bounding box of the specified dimensions. It may include indicators for"
        " periodical boundary conditions along the three cardinal directions each.")
        .def("forCuboid", &BoundingBoxRenderer::forCuboid)
        .def("setColor",  &BoundingBoxRenderer::setOption<BoundingBoxRenderer::Option::COLOR>,
            "Set the color of the bounding box");


    // CoordinateSystemRenderer
    py::class_<CoordinateSystemRenderer, RendererBase, std::shared_ptr<CoordinateSystemRenderer>>(m, "CoordinateSystemRenderer",
        "This Renderer draws a coordinate cross consisting of three arrows, colored according to the applied colormap.")
        .def(py::init<View&>())
        .def("setAxisLength", &CoordinateSystemRenderer::setOption<CoordinateSystemRenderer::Option::AXIS_LENGTH>)
        .def("setNormalize",  &CoordinateSystemRenderer::setOption<CoordinateSystemRenderer::Option::NORMALIZE>);


    // SurfaceRenderer
    py::class_<SurfaceRenderer, RendererBase, std::shared_ptr<SurfaceRenderer>>(m, "SurfaceRenderer",
        "This class is used to draw a 2D surface.")
        .def(py::init<View&, VectorField&>());


    // IsosurfaceRenderer
    py::class_<IsosurfaceRenderer, RendererBase, std::shared_ptr<IsosurfaceRenderer>>(m, "IsosurfaceRenderer",
        "This class is used to draw isosurfaces based on a set function and colormap.")
        .def(py::init<View&, VectorField&>())
        .def("setIsoValue",               &IsosurfaceRenderer::setOption<IsosurfaceRenderer::Option::ISOVALUE>)
        .def("setLightingImplementation", &IsosurfaceRenderer::setOption<IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>)
        .def("setValueFunction",          &IsosurfaceRenderer::setOption<IsosurfaceRenderer::Option::VALUE_FUNCTION>)
        .def("setFlipNormals",            &IsosurfaceRenderer::setOption<IsosurfaceRenderer::Option::FLIP_NORMALS>);


    // VectorSphereRenderer
    py::class_<VectorSphereRenderer, RendererBase, std::shared_ptr<VectorSphereRenderer>>(m, "VectorSphereRenderer",
        "This class is used to draw a sphere with points around it, each point representing the direction of one of"
        " the vectors in the vector field.")
        .def(py::init<View&, VectorField&>())
        .def("setPointSizeRange",           &VectorSphereRenderer::setOption<VectorSphereRenderer::Option::POINT_SIZE_RANGE>)
        .def("setInnerSphereRadius",        &VectorSphereRenderer::setOption<VectorSphereRenderer::Option::INNER_SPHERE_RADIUS>)
        .def("setUseSphereFakePerspective", &VectorSphereRenderer::setOption<VectorSphereRenderer::Option::USE_SPHERE_FAKE_PERSPECTIVE>);

    // DotRenderer
    py::class_<DotRenderer, RendererBase, std::shared_ptr<DotRenderer>>(m, "DotRenderer",
        "This class is used to draw dots at the positions of vectorfield, with colors corresponding to direction.")
        .def(py::init<View&, VectorField&>())
        .def("setDotRadius",  &DotRenderer::setOption<DotRenderer::Option::DOT_RADIUS>,
             "Set the radius of a dot")
        .def("setDotStyle",  &DotRenderer::setOption<DotRenderer::Option::DOT_STYLE>,
             "Set the style of a dot");
}
