#include "VFRendering/GlyphRenderer.hxx"

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/Utilities.hxx"

#include "shaders/glyphs.vert.glsl.hxx"
#include "shaders/glyphs.frag.glsl.hxx"

namespace VFRendering {
GlyphRenderer::GlyphRenderer(const View& view, const VectorField& vf) : VectorFieldRenderer(view, vf) {}

void GlyphRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glGenBuffers(1, &m_position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 0);

    glGenBuffers(1, &m_normal_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 0);

    glGenBuffers(1, &m_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    m_num_indices = 0;

    glGenBuffers(1, &m_instance_position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_instance_position_vbo);
    glVertexAttribPointer(2, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glGenBuffers(1, &m_instance_direction_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_instance_direction_vbo);
    glVertexAttribPointer(3, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);

    m_num_instances = 0;
    updateShaderProgram();
    if (m_indices.size() > 0) {
        setGlyph(m_positions, m_normals, m_indices);
    }
    update(false);
}

GlyphRenderer::~GlyphRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_position_vbo);
    glDeleteBuffers(1, &m_normal_vbo);
    glDeleteBuffers(1, &m_ibo);
    glDeleteBuffers(1, &m_instance_position_vbo);
    glDeleteBuffers(1, &m_instance_direction_vbo);
    glDeleteProgram(m_program);
}

void GlyphRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }
    bool update_shader = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::COLORMAP_IMPLEMENTATION:
        case View::Option::IS_VISIBLE_IMPLEMENTATION:
        case GlyphRenderer::Option::ROTATE_GLYPHS:
            update_shader = true;
            break;
        }
    }
    if (update_shader) {
        updateShaderProgram();
    }
}

void GlyphRenderer::update(bool keep_geometry) {
    if (!m_is_initialized) {
        return;
    }
    glBindVertexArray(m_vao);
    if (!keep_geometry) {
        glBindBuffer(GL_ARRAY_BUFFER, m_instance_position_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions().size(), positions().data(), GL_STREAM_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_instance_direction_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions().size(), directions().data(), GL_STREAM_DRAW);

    m_num_instances = std::min(positions().size(), directions().size());
}

void GlyphRenderer::draw(float aspect_ratio) {
    initialize();
    if (m_num_instances <= 0) {
        return;
    }
    glBindVertexArray(m_vao);
    glUseProgram(m_program);

    auto matrices = Utilities::getMatrices(options(), aspect_ratio);
    auto model_view_matrix = matrices.first;
    auto projection_matrix = matrices.second;
    glm::vec3 camera_position = options().get<View::Option::CAMERA_POSITION>();
    glm::vec3 light_position = options().get<View::Option::LIGHT_POSITION>();

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(model_view_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uLightPosition"), light_position[0], light_position[1], light_position[2]);

    glDisable(GL_CULL_FACE);
    glDrawElementsInstanced(GL_TRIANGLES, m_num_indices, GL_UNSIGNED_SHORT, nullptr, m_num_instances);
    glEnable(GL_CULL_FACE);
}

void GlyphRenderer::updateShaderProgram() {
    if (!m_is_initialized) {
        return;
    }
    if (m_program) {
        glDeleteProgram(m_program);
    }
    std::string vertex_shader_source;
    if (options().get<GlyphRenderer::Option::ROTATE_GLYPHS>()) {
        vertex_shader_source = GLYPHS_ROTATED_VERT_GLSL;
    } else {
        vertex_shader_source = GLYPHS_UNROTATED_VERT_GLSL;
    }
    vertex_shader_source += options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    vertex_shader_source += options().get<View::Option::IS_VISIBLE_IMPLEMENTATION>();
    std::string fragment_shader_source = GLYPHS_FRAG_GLSL;
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition", "ivNormal", "ivInstanceOffset", "ivInstanceDirection"});
}

void GlyphRenderer::setGlyph(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals, const std::vector<std::uint16_t>& indices) {
    if (!m_is_initialized) {
        m_positions = positions;
        m_normals = normals;
        m_indices = indices;
        return;
    }
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_position_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(), positions.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), normals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * indices.size(), indices.data(), GL_STATIC_DRAW);
    m_num_indices = indices.size();

    // Clear glyph data that might have been stored before OpenGL was ready
    m_positions.clear();
    m_normals.clear();
    m_indices.clear();
}
}
