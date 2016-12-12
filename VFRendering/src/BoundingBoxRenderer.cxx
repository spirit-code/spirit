#include "VFRendering/BoundingBoxRenderer.hxx"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/View.hxx"
#include "VFRendering/Utilities.hxx"

#include "shaders/boundingbox.vert.glsl.hxx"
#include "shaders/boundingbox.frag.glsl.hxx"

namespace VFRendering {
BoundingBoxRenderer::BoundingBoxRenderer(const View& view, const std::vector<glm::vec3>& vertices) : RendererBase(view), m_vertices(vertices) {}

void BoundingBoxRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);
    
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
    (void)changed_options;
}

void BoundingBoxRenderer::update(bool keep_geometry) {
    (void)keep_geometry;
}

void BoundingBoxRenderer::draw(float aspect_ratio) {
    initialize();

    if (m_vertices.size() == 0) {
        return;
    }

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
    glDrawArrays(GL_LINES, 0, m_vertices.size());
    glEnable(GL_CULL_FACE);
}

BoundingBoxRenderer BoundingBoxRenderer::forCuboid(const View& view, const glm::vec3& center, const glm::vec3& side_lengths) {
    glm::vec3 min = center - 0.5f*side_lengths;
    glm::vec3 max = center + 0.5f*side_lengths;
    std::vector<glm::vec3> bounding_box_vertices = {
        {min[0], min[1], min[2]}, {max[0], min[1], min[2]},
        {max[0], min[1], min[2]}, {max[0], max[1], min[2]},
        {max[0], max[1], min[2]}, {min[0], max[1], min[2]},
        {min[0], max[1], min[2]}, {min[0], min[1], min[2]},
        {min[0], min[1], max[2]}, {max[0], min[1], max[2]},
        {max[0], min[1], max[2]}, {max[0], max[1], max[2]},
        {max[0], max[1], max[2]}, {min[0], max[1], max[2]},
        {min[0], max[1], max[2]}, {min[0], min[1], max[2]},
        {min[0], min[1], min[2]}, {min[0], min[1], max[2]},
        {max[0], min[1], min[2]}, {max[0], min[1], max[2]},
        {max[0], max[1], min[2]}, {max[0], max[1], max[2]},
        {min[0], max[1], min[2]}, {min[0], max[1], max[2]}
    };
    return BoundingBoxRenderer(view, bounding_box_vertices);
}

BoundingBoxRenderer BoundingBoxRenderer::forParallelepiped(const View& view, const glm::vec3& center, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3) {
    glm::vec3 min = center - 0.5f*v1-0.5f*v2-0.5f*v3;
    std::vector<glm::vec3> bounding_box_vertices = {
        min, min+v1,
        min+v1, min+v1+v2,
        min+v1+v2, min+v2,
        min+v2, min,
        min+v3, min+v1+v3,
        min+v1+v3, min+v1+v2+v3,
        min+v1+v2+v3, min+v2+v3,
        min+v2+v3, min+v3,
        min, min+v3,
        min+v1, min+v1+v3,
        min+v2, min+v2+v3,
        min+v1+v2, min+v1+v2+v3
    };
    return BoundingBoxRenderer(view, bounding_box_vertices);
}

BoundingBoxRenderer BoundingBoxRenderer::forHexagonalCell(const View& view, const glm::vec3& center, float radius, float height) {
    glm::vec3 v1 = {radius, 0, 0};
    glm::vec3 v2 = {radius*0.5f, radius*0.8660254f, 0};
    glm::vec3 v3 = {0, 0, height/2};
    std::vector<glm::vec3> bounding_box_vertices = {
        center+v3+v1, center+v3+v2,
        center+v3+v2, center+v3+v2-v1,
        center+v3+v2-v1, center+v3-v1,
        center+v3-v1, center+v3-v2,
        center+v3-v2, center+v3+v1-v2,
        center+v3+v1-v2, center+v3+v1,
        center-v3+v1, center-v3+v2,
        center-v3+v2, center-v3+v2-v1,
        center-v3+v2-v1, center-v3-v1,
        center-v3-v1, center-v3-v2,
        center-v3-v2, center-v3+v1-v2,
        center-v3+v1-v2, center-v3+v1,
        center-v3+v1, center+v3+v1,
        center-v3+v2, center+v3+v2,
        center-v3+v2-v1, center+v3+v2-v1,
        center-v3-v1, center+v3-v1,
        center-v3-v2, center+v3-v2,
        center-v3+v1-v2, center+v3+v1-v2
    };
    return BoundingBoxRenderer(view, bounding_box_vertices);
}
}
