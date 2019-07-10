#include <iostream>
#include <GLES3/gl3.h>
#include <emscripten/html5.h>

#include "VFRendering/View.hxx"
#include "VFRendering/DotRenderer.hxx"
#include "VFRendering/ArrowRenderer.hxx"
#include "VFRendering/SphereRenderer.hxx"
#include "VFRendering/VectorSphereRenderer.hxx"
#include "VFRendering/CoordinateSystemRenderer.hxx"
#include "VFRendering/BoundingBoxRenderer.hxx"
#include "VFRendering/CombinedRenderer.hxx"
#include "VFRendering/IsosurfaceRenderer.hxx"

static bool needs_redraw = false;
VFRendering::View view;


void framebufferSizeCallback(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context, int width, int height) {
    (void)context;
    view.setFramebufferSize(width, height);
    needs_redraw = true;
}

extern "C" void display() {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion=1;
    attrs.minorVersion=0;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context = emscripten_webgl_create_context("#myCanvas", &attrs);
    emscripten_webgl_make_context_current(context);
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    int width, height;
    emscripten_webgl_get_drawing_buffer_size(context, &width, &height);
    framebufferSizeCallback(context, width, height);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> directions;
    for (int z = -20; z <= 20; z+=2) {
        for (int y = -20; y <= 20; y+=2) {
            for (int x = -20; x <= 20; x+=2) {
                positions.push_back({x, y, z});
                directions.push_back(glm::vec3{x, y, z} / glm::length(positions.back()));
            }
        }
    }
    VFRendering::Geometry geometry = VFRendering::Geometry::cartesianGeometry({21, 21, 21}, {-20, -20, -20}, {20, 20, 20});
    VFRendering::VectorField vf = VFRendering::VectorField(geometry, directions);


    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>((geometry.min() + geometry.max()) * 0.5f);
    options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    options.set<VFRendering::View::Option::CAMERA_POSITION>({-40, 40, 80});
    options.set<VFRendering::View::Option::CENTER_POSITION>({0, 0, 0});
    options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});
    view.updateOptions(options);

    auto isosurface_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, vf);
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
        (void)direction;
        return position.z * position.z + position.y * position.y + position.x * position.x - 19*19;
    });
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    auto yzplane_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, vf);
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
        (void)direction;
        return position.x;
    });
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    auto arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(view, vf);
    auto dot_renderer_ptr = std::make_shared<VFRendering::DotRenderer>(view, vf);
    auto sphere_renderer_ptr = std::make_shared<VFRendering::SphereRenderer>(view, vf);
    auto bounding_box_renderer_ptr = std::make_shared<VFRendering::BoundingBoxRenderer>(VFRendering::BoundingBoxRenderer::forCuboid(view, (geometry.min()+geometry.max())*0.5f, geometry.max()-geometry.min(), (geometry.max()-geometry.min())*0.2f, 0.5f));
    auto coordinate_system_renderer_ptr = std::make_shared<VFRendering::CoordinateSystemRenderer>(view);
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::AXIS_LENGTH>({0, 20, 20});
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>(false);

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
            isosurface_renderer_ptr,
            yzplane_renderer_ptr,
            arrow_renderer_ptr,
            sphere_renderer_ptr,
            dot_renderer_ptr,
            bounding_box_renderer_ptr,
            coordinate_system_renderer_ptr
    };
    view.renderers({{std::make_shared<VFRendering::CombinedRenderer>(view, renderers), {{0, 0, 1, 1}}}});
    view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x >= 0.0; }");
    arrow_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
    sphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return -normal.z; }");

    view.draw();
}