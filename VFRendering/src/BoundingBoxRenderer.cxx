#include "VFRendering/BoundingBoxRenderer.hxx"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/View.hxx"
#include "VFRendering/Utilities.hxx"

#include "shaders/boundingbox.vert.glsl.hxx"
#include "shaders/boundingbox.frag.glsl.hxx"

namespace VFRendering {
BoundingBoxRenderer::BoundingBoxRenderer(const View& view) : RendererBase(view) {}

void BoundingBoxRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);
    
    updateVertexData();
    
    std::string vertex_shader_source = BOUNDINGBOX_VERT_GLSL;
    std::string fragment_shader_source = BOUNDINGBOX_FRAG_GLSL;
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition"});
}

BoundingBoxRenderer::~BoundingBoxRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteProgram(m_program);
}

void BoundingBoxRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::BOUNDING_BOX_MIN:
        case View::Option::BOUNDING_BOX_MAX:
            update_vertices = true;
            break;
        }
    }
    if (update_vertices) {
        updateVertexData();
    }
}

void BoundingBoxRenderer::update(bool keep_geometry) {
    (void)keep_geometry;
}

void BoundingBoxRenderer::draw(float aspect_ratio) {
    initialize();
    
    glUseProgram(m_program);
    glBindVertexArray(m_vao);

    auto matrices = Utilities::getMatrices(options(), aspect_ratio);
    auto model_view_matrix = matrices.first;
    auto projection_matrix = matrices.second;
    glm::vec3 color = options().get<Option::COLOR>();

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(model_view_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uColor"), color.r, color.g, color.b);

    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_LINES, 0, 24);
    glEnable(GL_CULL_FACE);
}

void BoundingBoxRenderer::updateVertexData() {
    if (!m_is_initialized) {
        return;
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    auto min = options().get<View::Option::BOUNDING_BOX_MIN>();
    auto max = options().get<View::Option::BOUNDING_BOX_MAX>();
    std::vector<GLfloat> vertices = {
        min[0], min[1], min[2], max[0], min[1], min[2],
        max[0], min[1], min[2], max[0], max[1], min[2],
        max[0], max[1], min[2], min[0], max[1], min[2],
        min[0], max[1], min[2], min[0], min[1], min[2],
        min[0], min[1], max[2], max[0], min[1], max[2],
        max[0], min[1], max[2], max[0], max[1], max[2],
        max[0], max[1], max[2], min[0], max[1], max[2],
        min[0], max[1], max[2], min[0], min[1], max[2],
        min[0], min[1], min[2], min[0], min[1], max[2],
        max[0], min[1], min[2], max[0], min[1], max[2],
        max[0], max[1], min[2], max[0], max[1], max[2],
        min[0], max[1], min[2], min[0], max[1], max[2]
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
}
}
