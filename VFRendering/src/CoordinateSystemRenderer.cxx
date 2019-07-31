#include "VFRendering/CoordinateSystemRenderer.hxx"

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

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
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 4 * 3 * 3, nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 4 * 3 * 3, (void*)(4 * 3));
    glVertexAttribPointer(2, 3, GL_FLOAT, false, 4 * 3 * 3, (void*)(4 * 3 * 2));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    
    updateVertexData();
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
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::COLORMAP_IMPLEMENTATION:
            update_shader = true;
            break;
        case Option::AXIS_LENGTH:
        case Option::CONE_RADIUS:
        case Option::CONE_HEIGHT:
        case Option::CYLINDER_RADIUS:
        case Option::CYLINDER_HEIGHT:
        case Option::LEVEL_OF_DETAIL:
        case Option::NORMALIZE:
            update_vertices = true;
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

void CoordinateSystemRenderer::update(bool keep_geometry) {
    (void)keep_geometry;
}

void CoordinateSystemRenderer::draw(float aspect_ratio) {
    initialize();

    if (m_num_vertices <= 0) {
        return;
    }

    glm::vec3 camera_position = options().get<View::Option::CAMERA_POSITION>();
    glm::vec3 center_position = options().get<View::Option::CENTER_POSITION>();
    glm::vec3 up_vector = options().get<View::Option::UP_VECTOR>();
    glm::vec3 origin = options().get<Option::ORIGIN>();


    auto matrices = Utilities::getMatrices(options(), aspect_ratio);
    auto modelview_matrix = matrices.first;
    auto projection_matrix = matrices.second;

    if (options().get<Option::NORMALIZE>()) {
        if (options().get<View::Option::VERTICAL_FIELD_OF_VIEW>() == 0) {
            if (aspect_ratio > 1) {
                projection_matrix = glm::ortho(-0.5f*aspect_ratio, 0.5f*aspect_ratio, -0.5f, 0.5f, -10.0f, 10.0f);
            } else {
                projection_matrix = glm::ortho(-0.5f, 0.5f, -0.5f/aspect_ratio, 0.5f/aspect_ratio, -10.0f, 10.0f);
            }
        }
        modelview_matrix = glm::lookAt(glm::normalize(camera_position - center_position), glm::vec3(0.0, 0.0, 0.0), up_vector);
    }

    glUseProgram(m_program);
    glBindVertexArray(m_vao);

    glUniformMatrix4fv(glGetUniformLocation(m_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv(glGetUniformLocation(m_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelview_matrix));
    glUniform3f(glGetUniformLocation(m_program, "uOrigin"), origin.x, origin.y, origin.z);

    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_TRIANGLES, 0, m_num_vertices);
    glEnable(GL_CULL_FACE);
}

static std::vector<glm::vec3> generateSphereMesh(unsigned int subdivision_level=4) {
    float phi = 1.618033988749895;
    std::vector<glm::vec3> positions = {
        {-1, phi, 0},
        {1, phi, 0},
        {-1, -phi, 0},
        {1, -phi, 0},
        {0, -1, phi},
        {0, 1, phi},
        {0, -1, -phi},
        {0, 1, -phi},
        {phi, 0, -1},
        {phi, 0, 1},
        {-phi, 0, -1},
        {-phi, 0, 1}
    };
    std::vector<glm::ivec3> indices = {
        {0, 11, 5},
        {0, 5, 1},
        {0, 1, 7},
        {0, 7, 10},
        {0, 10, 11},
        {1, 5, 9},
        {5, 11, 4},
        {11, 10, 2},
        {10, 7, 6},
        {7, 1, 8},
        {3, 9, 4},
        {3, 4, 2},
        {3, 2, 6},
        {3, 6, 8},
        {3, 8, 9},
        {4, 9, 5},
        {2, 4, 11},
        {6, 2, 10},
        {8, 6, 7},
        {9, 8, 1}
    };
    std::vector<glm::vec3> vertices;
    for (const auto& triangle_indices : indices) {
        vertices.push_back(glm::normalize(positions[triangle_indices[0]]));
        vertices.push_back(glm::normalize(positions[triangle_indices[1]]));
        vertices.push_back(glm::normalize(positions[triangle_indices[2]]));
    }
    for (unsigned int j = 0; j < subdivision_level; j++) {
        std::vector<glm::vec3> previous_vertices;
        std::swap(vertices, previous_vertices);
        for (std::vector<glm::vec3>::size_type i = 0; i+2 < previous_vertices.size(); i += 3) {
            glm::vec3 a = previous_vertices[i];
            glm::vec3 b = previous_vertices[i+1];
            glm::vec3 c = previous_vertices[i+2];
            glm::vec3 ab = glm::normalize(a+b);
            glm::vec3 bc = glm::normalize(b+c);
            glm::vec3 ca = glm::normalize(c+a);
            vertices.push_back(a);
            vertices.push_back(ab);
            vertices.push_back(ca);
            vertices.push_back(ab);
            vertices.push_back(b);
            vertices.push_back(bc);
            vertices.push_back(bc);
            vertices.push_back(c);
            vertices.push_back(ca);
            vertices.push_back(ab);
            vertices.push_back(bc);
            vertices.push_back(ca);
        }
    }
    return vertices;
}

static std::vector<std::pair<glm::vec3, glm::vec3>> generateArrowMesh(unsigned int level_of_detail, float cone_height, float cone_radius, float cylinder_height, float cylinder_radius) {
    
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
    float z_offset = cylinder_height;
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
    
    std::vector<std::pair<glm::vec3, glm::vec3>> vertices;
    for (const auto& index : indices) {
        vertices.push_back({vertex_data[index*2], vertex_data[index*2+1]});
    }
    return vertices;
}

void CoordinateSystemRenderer::updateVertexData() {
    float rad90 = 1.5707963267948966f;
    glm::vec3 axis_length = options().get<Option::AXIS_LENGTH>();
    unsigned int level_of_detail = options().get<Option::LEVEL_OF_DETAIL>();
    float cone_height = options().get<Option::CONE_HEIGHT>();
    float cone_radius = options().get<Option::CONE_RADIUS>();
    float cylinder_height = options().get<Option::CYLINDER_HEIGHT>();
    float cylinder_radius = options().get<Option::CYLINDER_RADIUS>();
    float max_length = glm::max(axis_length.x, glm::max(axis_length.y, axis_length.z));
    if (options().get<Option::NORMALIZE>()) {
        axis_length /= max_length;
        float height = (cylinder_height+cone_height)/0.4;
        cylinder_height /= height;
        cone_height /= height;
        cylinder_radius /= height;
        cone_radius /= height;
    } else {
        cone_radius *= max_length;
        cylinder_radius *= max_length;
    }
    float sphere_radius = cylinder_radius;
    std::vector<glm::vec3> vertices;
    auto sphere_positions = generateSphereMesh();
    for (const auto& position : sphere_positions) {
        vertices.push_back(position*sphere_radius);
        vertices.push_back({0, 0, 0});
        vertices.push_back(position);
    }
    auto arrow_positions = generateArrowMesh(level_of_detail, cone_height, cone_radius, cylinder_height, cylinder_radius);
    if (axis_length.z > 0) {
        for (const auto& vertex : arrow_positions) {
            const auto& original_position = vertex.first;
            const auto& original_normal = vertex.second;
            glm::vec3 position = original_position*glm::vec3(1, 1, axis_length.z);
            auto normal = original_normal;
            if (normal.z != 0) {
                normal *= glm::vec3(1, 1, 1/axis_length.z);
                glm::normalize(normal);
            }
            vertices.push_back(position);
            vertices.push_back({0, 0, 1});
            vertices.push_back(normal);
        }
    }
    if (axis_length.y > 0) {
        for (const auto& vertex : arrow_positions) {
            const auto& original_position = vertex.first;
            const auto& original_normal = vertex.second;
            glm::vec3 position = glm::rotate(original_position*glm::vec3(1, 1, axis_length.y), -rad90, {1, 0, 0});
            auto normal = original_normal;
            if (normal.z != 0) {
                normal *= glm::vec3(1, 1, 1/axis_length.y);
                glm::normalize(normal);
            }
            normal = glm::rotate(normal, -rad90, {1, 0, 0});
            vertices.push_back(position);
            vertices.push_back({0, 1, 0});
            vertices.push_back(normal);
        }
    }
    if (axis_length.x > 0) {
        for (const auto& vertex : arrow_positions) {
            const auto& original_position = vertex.first;
            const auto& original_normal = vertex.second;
            glm::vec3 position = glm::rotate(original_position*glm::vec3(1, 1, axis_length.x), rad90, {0, 1, 0});
            auto normal = original_normal;
            if (normal.z != 0) {
                normal *= glm::vec3(1, 1, 1/axis_length.x);
                glm::normalize(normal);
            }
            normal = glm::rotate(normal, rad90, {0, 1, 0});
            vertices.push_back(position);
            vertices.push_back({1, 0, 0});
            vertices.push_back(normal);
        }
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    m_num_vertices = vertices.size()/3;
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
