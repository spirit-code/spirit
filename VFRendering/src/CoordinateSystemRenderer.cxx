#include "VFRendering/CoordinateSystemRenderer.hxx"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "VFRendering/Utilities.hxx"

#include "shaders/coordinatesystem.vert.glsl.hxx"
#include "shaders/coordinatesystem.frag.glsl.hxx"

namespace VFRendering {
CoordinateSystemRenderer::CoordinateSystemRenderer(const View& view) : RendererBase(view) {}

void CoordinateSystemRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    std::vector<GLfloat> vertices = {
        0, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 1
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 4 * 3 * 2, nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 4 * 3 * 2, (void*)(4 * 3));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    
    updateShaderProgram();
}

CoordinateSystemRenderer::~CoordinateSystemRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteProgram(m_program);
}

void CoordinateSystemRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }
    bool update_shader = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::COLORMAP_IMPLEMENTATION:
            update_shader = true;
            break;
        }
    }
    if (update_shader) {
        updateShaderProgram();
    }
}

void CoordinateSystemRenderer::update(bool keep_geometry) {
    (void)keep_geometry;
}

void CoordinateSystemRenderer::draw(float aspect_ratio) {
    initialize();
    glm::vec3 camera_position = options().get<View::Option::CAMERA_POSITION>();
    glm::vec3 center_position = options().get<View::Option::CENTER_POSITION>();
    glm::vec3 up_vector = options().get<View::Option::UP_VECTOR>();
    glm::vec3 origin = options().get<Option::ORIGIN>();
    glm::vec3 axis_length = glm::normalize(options().get<Option::AXIS_LENGTH>());

    glm::mat4 modelview_matrix = glm::lookAt(glm::normalize(camera_position - center_position), glm::vec3(0.0, 0.0, 0.0), up_vector);

    auto matrices = Utilities::getMatrices(options(), aspect_ratio);
    auto projection_matrix = matrices.second;

    glUseProgram(m_program);
    glBindVertexArray(m_vao);

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelview_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uOrigin"), origin.x, origin.y, origin.z);
    glUniform3f(glGetUniformLocation(m_program, "uAxisLength"), axis_length.x, axis_length.y, axis_length.z);

    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_LINES, 0, 6);
    glEnable(GL_CULL_FACE);
}

void CoordinateSystemRenderer::updateShaderProgram() {
    if (!m_is_initialized) {
        return;
    }
    if (m_program) {
        glDeleteProgram(m_program);
    }
    m_program = 0;

    std::string vertex_shader_source = COORDINATESYSTEM_VERT_GLSL;
    vertex_shader_source += options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    std::string fragment_shader_source = COORDINATESYSTEM_FRAG_GLSL;
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition", "ivDirection"});
}
}
