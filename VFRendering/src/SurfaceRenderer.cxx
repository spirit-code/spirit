#include "VFRendering/SurfaceRenderer.hxx"

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/Utilities.hxx"

#include "shaders/surface.vert.glsl.hxx"
#include "shaders/surface.frag.glsl.hxx"

namespace VFRendering {
SurfaceRenderer::SurfaceRenderer(const View& view, const VectorField& vf) : VectorFieldRenderer(view, vf) {}

void SurfaceRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    
    glGenBuffers(1, &m_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    m_num_indices = 0;

    glGenBuffers(1, &m_position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &m_direction_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_direction_vbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(1);

    updateShaderProgram();
    update(false);
}

SurfaceRenderer::~SurfaceRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_ibo);
    glDeleteBuffers(1, &m_position_vbo);
    glDeleteBuffers(1, &m_direction_vbo);
    glDeleteProgram(m_program);
}

void SurfaceRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }
    bool update_shader = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::COLORMAP_IMPLEMENTATION:
        case View::Option::IS_VISIBLE_IMPLEMENTATION:
            update_shader = true;
            break;
        }
    }
    if (update_shader) {
        updateShaderProgram();
    }
}

void SurfaceRenderer::update(bool keep_geometry) {
    if (!m_is_initialized) {
        return;
    }
    glBindVertexArray(m_vao);
    if (!keep_geometry) {
        glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions().size(), positions().data(), GL_STREAM_DRAW);
        updateSurfaceIndices();
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_direction_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions().size(), directions().data(), GL_STREAM_DRAW);
}

void SurfaceRenderer::draw(float aspect_ratio) {
    initialize();
    if (m_num_indices <= 0) {
        return;
    }
    glBindVertexArray(m_vao);
    glUseProgram(m_program);

    auto matrices = Utilities::getMatrices(options(), aspect_ratio);
    auto model_view_matrix = matrices.first;
    auto projection_matrix = matrices.second;
    glm::vec3 camera_position = options().get<View::Option::CAMERA_POSITION>();
    glm::vec4 light_position = model_view_matrix * glm::vec4(camera_position, 1.0f);

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(model_view_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uLightPosition"), light_position[0], light_position[1], light_position[2]);

    glDisable(GL_CULL_FACE);
    glDrawElements(GL_TRIANGLES, m_num_indices, GL_UNSIGNED_INT, nullptr);
    glEnable(GL_CULL_FACE);
}

void SurfaceRenderer::updateShaderProgram() {
    if (!m_is_initialized) {
        return;
    }
    if (m_program) {
        glDeleteProgram(m_program);
    }
    std::string vertex_shader_source = SURFACE_VERT_GLSL;
    vertex_shader_source += options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    std::string fragment_shader_source = SURFACE_FRAG_GLSL;
    fragment_shader_source += options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    fragment_shader_source += options().get<View::Option::IS_VISIBLE_IMPLEMENTATION>();
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition", "ivDirection"});
}

void SurfaceRenderer::updateSurfaceIndices() {
    if (!m_is_initialized) {
        return;
    }
    const auto& surface_indices = surfaceIndices();
    if (surface_indices.empty()) {
        m_num_indices = 0;
        return;
    }
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint) * surface_indices.size(), &(surface_indices[0][0]), GL_STREAM_DRAW);
    m_num_indices = 3 * surface_indices.size();
}
}
