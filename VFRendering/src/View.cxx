#include "VFRendering/View.hxx"

#include <iostream>

#include <glm/gtx/transform.hpp>
#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include "VFRendering/RendererBase.hxx"
#include "VFRendering/ArrowRenderer.hxx"
#include "VFRendering/SurfaceRenderer.hxx"
#include "VFRendering/IsosurfaceRenderer.hxx"
#include "VFRendering/VectorSphereRenderer.hxx"
#include "VFRendering/BoundingBoxRenderer.hxx"
#include "VFRendering/CombinedRenderer.hxx"
#include "VFRendering/CoordinateSystemRenderer.hxx"

namespace VFRendering {
View::View() { }

void View::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

#ifndef EMSCRIPTEN
#ifndef SPIRIT_UI_USE_IMGUI
    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize glad" << std::endl;
        return;
    }
#endif
#endif
    // Reset any errors potentially caused by the extension loader
    glGetError();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}

View::~View() {}

void View::draw() {
    initialize();
    if (m_options.get<View::Option::CLEAR>()) {
        auto background_color = m_options.get<View::Option::BACKGROUND_COLOR>();
        glClearColor(background_color.x, background_color.y, background_color.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    for (auto it : m_renderers) {
        auto renderer = it.first;
        auto viewport = it.second;
        float width = m_framebuffer_size.x;
        float height = m_framebuffer_size.y;
        glViewport((GLint)(viewport[0] * width), (GLint)(viewport[1] * height), (GLsizei)(viewport[2] * width), (GLsizei)(viewport[3] * height));
        if (m_options.get<View::Option::CLEAR>()) {
            glClear(GL_DEPTH_BUFFER_BIT);
        }
        renderer->updateIfNecessary();
        renderer->draw(viewport[2] * width / viewport[3] / height);
    }
    m_fps_counter.tick();
}

float View::getFramerate() const {
    return m_fps_counter.getFramerate();
}

void View::mouseMove(const glm::vec2& position_before, const glm::vec2& position_after, CameraMovementModes mode) {
    if (position_before == position_after) {
        return;
    }
    auto camera_position = options().get<Option::CAMERA_POSITION>();
    auto center_position = options().get<Option::CENTER_POSITION>();
    auto up_vector = options().get<Option::UP_VECTOR>();
    auto delta = position_after - position_before;
    auto length = glm::length(delta);
    auto forward = glm::normalize(center_position - camera_position);
    auto camera_distance = glm::length(center_position - camera_position);
    up_vector = glm::normalize(up_vector);
    auto right = glm::normalize(glm::cross(forward, up_vector));
    up_vector = glm::normalize(glm::cross(right, forward));
    delta = glm::normalize(delta);
    switch (mode) {
    case CameraMovementModes::ROTATE_FREE: {
        auto axis = glm::normalize(delta.x * up_vector + delta.y * right);
        float angle = -length * 0.1f / 180 * 3.14f;
        auto rotation_matrix = glm::rotate(angle, axis);
        up_vector = glm::mat3(rotation_matrix) * up_vector;
        forward = glm::mat3(rotation_matrix) * forward;
        bool was_centered = m_is_centered;
        setCamera(center_position - forward * camera_distance, center_position, up_vector);
        m_is_centered = was_centered;
    }
    break;
    case CameraMovementModes::ROTATE_BOUNDED: {
        auto zaxis = glm::vec3{0,0,1};
        // Get correct right-vector (in xy-plane) and orthogonal right and up
        right = right - zaxis * glm::dot(right, zaxis);
        glm::normalize(right);
        up_vector = up_vector - right*glm::dot(up_vector, right);
        glm::normalize(up_vector);
        forward = forward - right*glm::dot(forward, right);
        glm::normalize(forward);
        // Get rotation axis & angle
        auto axis = glm::normalize(delta.x * zaxis + delta.y * right);
        float angle = -length * 0.1f / 180 * 3.14f;
        // Rotate up and forward vectors
        auto rotation_matrix = glm::rotate(angle, axis);
        up_vector = glm::mat3(rotation_matrix) * up_vector;
        forward = glm::mat3(rotation_matrix) * forward;
        // Set new camera position
        bool was_centered = m_is_centered;
        setCamera(center_position - forward * camera_distance, center_position, up_vector);
        m_is_centered = was_centered;
    }
    break;
    case CameraMovementModes::TRANSLATE: {
        float factor = 0.001f * camera_distance * length;
        auto translation = factor * (delta.y * up_vector - delta.x * right);
        setCamera(camera_position + translation, center_position + translation, up_vector);
        m_is_centered = false;
    }
    break;
    default:
        break;
    }
}

void View::mouseScroll(const float& wheel_delta) {
    auto camera_position = options().get<Option::CAMERA_POSITION>();
    auto center_position = options().get<Option::CENTER_POSITION>();
    auto up_vector = options().get<Option::UP_VECTOR>();
    auto forward = center_position - camera_position;
    float camera_distance = glm::length(forward);
    float new_camera_distance = (float)(1 + 0.02 * wheel_delta) * camera_distance;
    float min_camera_distance = 1;
    if (new_camera_distance < min_camera_distance) {
        new_camera_distance = min_camera_distance;
    }

    camera_position = center_position - new_camera_distance / camera_distance * forward;
    setCamera(camera_position, center_position, up_vector);
}

void View::setFramebufferSize(float width, float height) {
    m_framebuffer_size = glm::vec2(width, height);
}

glm::vec2 View::getFramebufferSize() const {
    return m_framebuffer_size;
}

void View::setCamera(glm::vec3 camera_position, glm::vec3 center_position, glm::vec3 up_vector) {
    Options options;
    options.set<Option::CAMERA_POSITION>(camera_position);
    options.set<Option::CENTER_POSITION>(center_position);
    options.set<Option::UP_VECTOR>(up_vector);
    updateOptions(options);
    m_is_centered = false;
}

void View::renderers(const std::vector<std::pair<std::shared_ptr<RendererBase>, std::array<float, 4>>>& renderers, bool update_renderer_options) {
    m_renderers = renderers;
    if (update_renderer_options) {
        for (auto it : m_renderers) {
            auto renderer = it.first;
            renderer->updateOptions(options());
        }
    }
}

void View::updateOptions(const Options& options) {
    auto changed_options = m_options.update(options);
    if (changed_options.size() == 0) {
        return;
    }
    optionsHaveChanged(changed_options);
    for (auto it : m_renderers) {
        auto renderer = it.first;
        renderer->updateOptions(options);
    }
}

void View::options(const Options& options) {
    m_options = Options();
    updateOptions(options);
}

const Options& View::options() const {
    return m_options;
}

void View::optionsHaveChanged(const std::vector<int>& changed_options) {
    bool recenter_camera = false;
    for (int option_index : changed_options) {
        switch (option_index) {
        case Option::SYSTEM_CENTER:
            recenter_camera = true;
            break;
        case Option::CENTER_POSITION:
            if (options().get<Option::CENTER_POSITION>() == options().get<Option::SYSTEM_CENTER>()) {
                m_is_centered = true;
            }
            break;
        }
    }
    if (m_is_centered && recenter_camera) {
        auto camera_position = options().get<Option::CAMERA_POSITION>();
        auto center_position = options().get<Option::CENTER_POSITION>();
        auto up_vector = options().get<Option::UP_VECTOR>();
        camera_position = camera_position + options().get<Option::SYSTEM_CENTER>() - center_position;
        center_position = options().get<Option::SYSTEM_CENTER>();
        setCamera(camera_position, center_position, up_vector);
    }
}
}
