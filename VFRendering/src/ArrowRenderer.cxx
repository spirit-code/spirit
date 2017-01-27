#include "VFRendering/ArrowRenderer.hxx"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/Utilities.hxx"

#include "shaders/arrows.vert.glsl.hxx"
#include "shaders/arrows.frag.glsl.hxx"

namespace VFRendering {
ArrowRenderer::ArrowRenderer(const View& view) : RendererBase(view) {}

void ArrowRenderer::initialize() {
    if (m_is_initialized) {
        return;
    }
    m_is_initialized = true;
    
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 4 * 3 * 2, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 4 * 3 * 2, (void*)(4 * 3));
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
    updateVertexData();
    update(false);
}

ArrowRenderer::~ArrowRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ibo);
    glDeleteBuffers(1, &m_instance_position_vbo);
    glDeleteBuffers(1, &m_instance_direction_vbo);
    glDeleteProgram(m_program);
}

void ArrowRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }
    bool update_shader = false;
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case Option::CONE_RADIUS:
        case Option::CONE_HEIGHT:
        case Option::CYLINDER_RADIUS:
        case Option::CYLINDER_HEIGHT:
        case Option::LEVEL_OF_DETAIL:
            update_vertices = true;
            break;
        case View::Option::COLORMAP_IMPLEMENTATION:
        case View::Option::IS_VISIBLE_IMPLEMENTATION:
            update_shader = true;
            break;
        }
    }
    if (update_shader) {
        updateShaderProgram();
    }
    if (update_vertices) {
        updateVertexData();
    }
}

void ArrowRenderer::update(bool keep_geometry) {
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

void ArrowRenderer::draw(float aspect_ratio) {
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
    glm::vec4 light_position = model_view_matrix * glm::vec4(camera_position, 1.0);

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(model_view_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uLightPosition"), light_position[0], light_position[1], light_position[2]);

    glDisable(GL_CULL_FACE);
    glDrawElementsInstanced(GL_TRIANGLES, m_num_indices, GL_UNSIGNED_SHORT, nullptr, m_num_instances);
    glEnable(GL_CULL_FACE);
}

void ArrowRenderer::updateShaderProgram() {
    if (!m_is_initialized) {
        return;
    }
    if (m_program) {
        glDeleteProgram(m_program);
    }
    std::string vertex_shader_source = ARROWS_VERT_GLSL;
    vertex_shader_source += options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    vertex_shader_source += options().get<View::Option::IS_VISIBLE_IMPLEMENTATION>();
    std::string fragment_shader_source = ARROWS_FRAG_GLSL;
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition", "ivNormal", "ivInstanceOffset", "ivInstanceDirection"});
}

void ArrowRenderer::updateVertexData() {
    if (!m_is_initialized) {
        return;
    }
    auto level_of_detail = options().get<Option::LEVEL_OF_DETAIL>();
    auto cone_height = options().get<Option::CONE_HEIGHT>();
    auto cone_radius = options().get<Option::CONE_RADIUS>();
    auto cylinder_height = options().get<Option::CYLINDER_HEIGHT>();
    auto cylinder_radius = options().get<Option::CYLINDER_RADIUS>();

    // Enforce valid range
    if (level_of_detail < 3) {
        level_of_detail = 3;
    }
    if (cone_height < 0) {
        cone_height = 0;
    }
    if (cone_radius < 0) {
        cone_radius = 0;
    }
    if (cylinder_height < 0) {
        cylinder_height = 0;
    }
    if (cylinder_radius < 0) {
        cylinder_radius = 0;
    }
    unsigned int i;
    double pi = 3.14159265358979323846;
    glm::vec3 baseNormal = {0, 0, -1};
    float z_offset = (cylinder_height - cone_height) / 2;
    float l = sqrt(cone_radius * cone_radius + cone_height * cone_height);
    float f1 = cone_radius / l;
    float f2 = cone_height / l;
    std::vector<glm::vec3> vertex_data;
    vertex_data.reserve(level_of_detail * 5 * 2);
    // The tip has no normal to prevent a discontinuity.
    vertex_data.push_back({0, 0, z_offset + cone_height});
    vertex_data.push_back({0, 0, 0});
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        vertex_data.push_back({cone_radius* cos(alpha), cone_radius * sin(alpha), z_offset});
        vertex_data.push_back({f2* cos(alpha), f2 * sin(alpha), f1});
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        vertex_data.push_back({cone_radius* cos(alpha), cone_radius * sin(alpha), z_offset});
        vertex_data.push_back(baseNormal);
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        vertex_data.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset - cylinder_height});
        vertex_data.push_back(baseNormal);
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        vertex_data.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset - cylinder_height});
        vertex_data.push_back({cos(alpha), sin(alpha), 0});
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        vertex_data.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset});
        vertex_data.push_back({cos(alpha), sin(alpha), 0});
    }
    std::vector<GLushort> indices;
    indices.reserve(level_of_detail * 15);
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(1 + i);
        indices.push_back(1 + (i + 1) % level_of_detail);
        indices.push_back(0);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail + 1);
        indices.push_back(level_of_detail + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail + 1 + i);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail * 2 + 1);
        indices.push_back(level_of_detail * 2 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 2 + 1 + i);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail * 3 + 1 + i);
        indices.push_back(level_of_detail * 3 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 4 + 1 + i);
        indices.push_back(level_of_detail * 4 + 1 + i);
        indices.push_back(level_of_detail * 3 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 4 + 1 + (i + 1) % level_of_detail);
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertex_data.size(), vertex_data.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * indices.size(), indices.data(), GL_STATIC_DRAW);
    m_num_indices = indices.size();
}
}
