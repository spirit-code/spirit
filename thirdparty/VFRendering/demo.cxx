#include <iostream>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "VFRendering/View.hxx"
#include "VFRendering/ArrowRenderer.hxx"
#include "VFRendering/SphereRenderer.hxx"
#include "VFRendering/CoordinateSystemRenderer.hxx"
#include "VFRendering/BoundingBoxRenderer.hxx"
#include "VFRendering/CombinedRenderer.hxx"
#include "VFRendering/IsosurfaceRenderer.hxx"

static bool needs_redraw = false;
VFRendering::View view;

void mouseWheelCallback(GLFWwindow* window, double x_offset, double y_offset) {
    (void)window;
    (void)x_offset;
    view.mouseScroll(y_offset);
    needs_redraw = true;
}

void mousePositionCallback(GLFWwindow* window, double x_position, double y_position) {
    static glm::vec2 previous_mouse_position(0, 0);
    glm::vec2 current_mouse_position(x_position, y_position);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        auto movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
        }
        view.mouseMove(previous_mouse_position, current_mouse_position, movement_mode);
        needs_redraw = true;
    }
    previous_mouse_position = current_mouse_position;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    (void)window;
    view.setFramebufferSize(width, height);
    needs_redraw = true;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    if (mods != 0) {
        return;
    }
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }
    switch (key) {
    case GLFW_KEY_R:
        {
            VFRendering::Options options;
            options.set<VFRendering::View::Option::CAMERA_POSITION>({0, 0, 30});
            options.set<VFRendering::View::Option::CENTER_POSITION>({0, 0, 0});
            options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});
            view.updateOptions(options);
        }
        needs_redraw = true;
        break;
    }
}

void windowRefreshCallback(GLFWwindow* window) {
    (void)window;
    needs_redraw = true;
}

int main(void) {
    GLFWwindow* window;
    if (!glfwInit()) {
        return -1;
    }
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    glfwWindowHint(GLFW_SAMPLES, 16);
    window = glfwCreateWindow(800, 800, "Demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetScrollCallback(window, mouseWheelCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize glad" << std::endl;
        return 1;
    }
    glEnable(GL_MULTISAMPLE);

    framebufferSizeCallback(window, 800, 800);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> directions;
    {
        std::ifstream f("data.txt");
        while (f.good()) {
            glm::vec3 position;
            glm::vec3 direction;
            f >> position.x >> position.y >> position.z >> direction.x >> direction.y >> direction.z;
            if (f.good()) {
                positions.push_back(position * 10.0f);
                directions.push_back(direction);
            }
        }
    }
    VFRendering::Geometry geometry = VFRendering::Geometry::cartesianGeometry({21, 21, 21}, {-20, -20, -20}, {20, 20, 20});
    VFRendering::VectorField vf = VFRendering::VectorField(geometry, directions);

    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>((geometry.min() + geometry.max()) * 0.5f);
    options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    options.set<VFRendering::View::Option::CAMERA_POSITION>({0, 0, 30});
    options.set<VFRendering::View::Option::CENTER_POSITION>({0, 0, 0});
    options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});
    view.updateOptions(options);

    auto isosurface_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, vf);
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
        (void)position;
        return direction.z;
    });
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    auto yzplane_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, vf);
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>([] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
        (void)direction;
        return position.x;
    });
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    auto arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(view, vf);
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
        bounding_box_renderer_ptr,
        coordinate_system_renderer_ptr
    };
    view.renderers({{std::make_shared<VFRendering::CombinedRenderer>(view, renderers), {{0, 0, 1, 1}}}});
    view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x >= 0; }");
    arrow_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x <= 0; }");
    sphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>("bool is_visible(vec3 position, vec3 direction) { return position.x <= 0; }");
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>("float lighting(vec3 position, vec3 normal) { return -normal.z; }");

    while (!glfwWindowShouldClose(window)) {
        if (needs_redraw) {
            needs_redraw = false;
            view.draw();
            glfwSwapBuffers(window);
        }
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
