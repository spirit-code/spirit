#include "VFRendering/BoundingBoxRenderer.hxx"

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/View.hxx"
#include "VFRendering/Utilities.hxx"

#include "shaders/boundingbox.vert.glsl.hxx"
#include "shaders/boundingbox.frag.glsl.hxx"

namespace VFRendering {
BoundingBoxRenderer::BoundingBoxRenderer(const View& view, const std::vector<glm::vec3>& vertices, const std::vector<float>& dashing_values) : RendererBase(view), m_vertices(vertices), m_dashing_values(dashing_values) {
    if (m_dashing_values.size() != m_vertices.size()) {
        m_dashing_values.resize(m_vertices.size(), 0);
    }
}

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

    glGenBuffers(1, &m_dash_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_dash_vbo);
    glVertexAttribPointer(1, 1, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(1);

    updateVertexData();

    std::string vertex_shader_source = BOUNDINGBOX_VERT_GLSL;
    std::string fragment_shader_source = BOUNDINGBOX_FRAG_GLSL;
    m_program = Utilities::createProgram(vertex_shader_source, fragment_shader_source, {"ivPosition", "ivDashingValue"});
}

BoundingBoxRenderer::~BoundingBoxRenderer() {
    if (!m_is_initialized) {
        return;
    }
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_dash_vbo);
    glDeleteProgram(m_program);
}

void BoundingBoxRenderer::updateVertexData() {
    float line_width = options().get<Option::LINE_WIDTH>();

    if (line_width <= 0.0) {
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, m_dash_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * m_dashing_values.size(), m_dashing_values.data(), GL_STATIC_DRAW);
    } else {
        float radius = line_width * 0.5;
        int level_of_detail = options().get<Option::LEVEL_OF_DETAIL>();
        if (level_of_detail < 3) {
            level_of_detail = 3;
        }

        std::vector <glm::vec3> cylinder_vertices;
        std::vector<float> cylinder_dashing_values;
        for (std::size_t i = 0; i + 1 < m_vertices.size(); i += 2) {
            glm::vec3 start = m_vertices[i];
            glm::vec3 end = m_vertices[i + 1];
            glm::vec3 direction = end - start;
            glm::vec3 ortho_x = glm::cross(direction, glm::vec3(1, 0, 0));
            glm::vec3 ortho_y = glm::cross(direction, glm::vec3(0, 1, 0));
            glm::vec3 ortho_z = glm::cross(direction, glm::vec3(0, 0, 1));
            float ortho_x_length = glm::length(ortho_x);
            float ortho_y_length = glm::length(ortho_y);
            float ortho_z_length = glm::length(ortho_z);
            glm::vec3 normal;
            if (ortho_x_length > ortho_y_length && ortho_x_length > ortho_z_length) {
                normal = ortho_x;
            } else if (ortho_y_length > ortho_z_length) {
                normal = ortho_y;
            } else {
                normal = ortho_z;
            }
            normal = radius * glm::normalize(normal);
            glm::vec3 binormal = radius * glm::normalize(glm::cross(direction, normal));

            for (int j = 0; j < level_of_detail; j++) {
                float start_angle = glm::radians(360.0 * j / level_of_detail);
                float end_angle = glm::radians(360.0 * (j + 1) / level_of_detail);

                cylinder_vertices.push_back(start + normal * glm::cos(start_angle) + binormal * glm::sin(start_angle));
                cylinder_vertices.push_back(start + normal * glm::cos(end_angle) + binormal * glm::sin(end_angle));
                cylinder_vertices.push_back(end + normal * glm::cos(start_angle) + binormal * glm::sin(start_angle));
                cylinder_vertices.push_back(end + normal * glm::cos(start_angle) + binormal * glm::sin(start_angle));
                cylinder_vertices.push_back(start + normal * glm::cos(end_angle) + binormal * glm::sin(end_angle));
                cylinder_vertices.push_back(end + normal * glm::cos(end_angle) + binormal * glm::sin(end_angle));
                cylinder_dashing_values.push_back(m_dashing_values[i]);
                cylinder_dashing_values.push_back(m_dashing_values[i]);
                cylinder_dashing_values.push_back(m_dashing_values[i + 1]);
                cylinder_dashing_values.push_back(m_dashing_values[i + 1]);
                cylinder_dashing_values.push_back(m_dashing_values[i]);
                cylinder_dashing_values.push_back(m_dashing_values[i + 1]);

                int first_dash = glm::floor(glm::min(m_dashing_values[i], m_dashing_values[i + 1]));
                int last_dash = glm::ceil(glm::max(m_dashing_values[i], m_dashing_values[i + 1]));
                for (int k = first_dash - 1; k < last_dash; k++) {
                    float d = k + 0.5;
                    if (glm::mod(glm::floor(d), 2.0f) != 0.0f) {
                        continue;
                    }
                    if (d < glm::min(m_dashing_values[i], m_dashing_values[i + 1]) || d > glm::max(m_dashing_values[i], m_dashing_values[i + 1])) {
                        continue;
                    }
                    float f = (d - m_dashing_values[i]) / (m_dashing_values[i + 1] - m_dashing_values[i]);
                    cylinder_vertices.push_back(start + direction * f);
                    cylinder_vertices.push_back(start + direction * f + normal * glm::cos(start_angle) + binormal * glm::sin(start_angle));
                    cylinder_vertices.push_back(start + direction * f + normal * glm::cos(end_angle) + binormal * glm::sin(end_angle));
                    cylinder_dashing_values.push_back(d);
                    cylinder_dashing_values.push_back(d);
                    cylinder_dashing_values.push_back(d);
                }
            }
        }

        num_vertices = cylinder_vertices.size();

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * cylinder_vertices.size(), cylinder_vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, m_dash_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * cylinder_dashing_values.size(), cylinder_dashing_values.data(), GL_STATIC_DRAW);
    }
}

void BoundingBoxRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    if (!m_is_initialized) {
        return;
    }

    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
            case Option::LINE_WIDTH:
            case Option::LEVEL_OF_DETAIL:
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
    if (options().get<Option::LINE_WIDTH>() <= 0.0) {
        glDrawArrays(GL_LINES, 0, m_vertices.size());
    } else {
        int level_of_detail = options().get<Option::LEVEL_OF_DETAIL>();
        if (level_of_detail < 3) {
            level_of_detail = 3;
        }
        glDrawArrays(GL_TRIANGLES, 0, num_vertices);
    }
    glEnable(GL_CULL_FACE);
}

BoundingBoxRenderer BoundingBoxRenderer::forCuboid(const View& view, const glm::vec3& center, const glm::vec3& side_lengths, const glm::vec3& periodic_boundary_condition_lengths, float dashes_per_length) {
    return BoundingBoxRenderer::forParallelepiped(view, center, {side_lengths.x, 0, 0}, {0, side_lengths.y, 0}, {0, 0, side_lengths.z}, periodic_boundary_condition_lengths, dashes_per_length);
}

BoundingBoxRenderer BoundingBoxRenderer::forParallelepiped(const View& view, const glm::vec3& center, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const glm::vec3& periodic_boundary_condition_lengths, float dashes_per_length) {
    std::vector<glm::vec3> bounding_box_vertices;
    std::vector<std::pair<glm::vec3, glm::ivec3>> points;
    for (int dx = -1; dx <= 1; dx += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dz = -1; dz <= 1; dz += 2) {
                glm::ivec3 ipoint = {dx, dy, dz};
                glm::vec3 point = center + dx*0.5f*v1+dy*0.5f*v2+dz*0.5f*v3;
                points.push_back({point, ipoint});
            }
        }
    }

    std::vector<int> edge_indices = {0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7};
    for (int index : edge_indices) {
        bounding_box_vertices.push_back(points[index].first);
    }

    std::vector<float> dashing_values;
    std::array<glm::vec3, 3> translation_vectors({{v1, v2, v3}});
    dashing_values.resize(bounding_box_vertices.size(), 0);
    for (int i = 0; i < 3; i++) {
        if (periodic_boundary_condition_lengths[i] > 0) {
            for (const auto& point : points) {
                bounding_box_vertices.push_back(point.first);
                dashing_values.push_back(0);
                bounding_box_vertices.push_back(point.first + point.second[i]*periodic_boundary_condition_lengths[i]*glm::normalize(translation_vectors[i]));
                dashing_values.push_back(dashes_per_length*periodic_boundary_condition_lengths[i]*2);
            }
        }
    }
    return BoundingBoxRenderer(view, bounding_box_vertices, dashing_values);
}

BoundingBoxRenderer BoundingBoxRenderer::forHexagonalCell(const View& view, const glm::vec3& center, float radius, float height, const glm::vec2& periodic_boundary_condition_lengths, float dashes_per_length) {
    glm::vec3 v1 = {radius, 0, 0};
    glm::vec3 v2 = {radius*0.5f, radius*0.8660254f, 0};
    glm::vec3 v3 = {0, 0, height/2};
    std::vector<glm::vec3> points = {
        center+v3+v1,
        center+v3+v2,
        center+v3+v2-v1,
        center+v3-v1,
        center+v3-v2,
        center+v3-v2+v1,
        center-v3+v1,
        center-v3+v2,
        center-v3+v2-v1,
        center-v3-v1,
        center-v3-v2,
        center-v3-v2+v1
    };

    std::vector<glm::vec3> bounding_box_vertices;
    std::vector<float> dashing_values;
    for (int i = 0; i < 6; i++) {
        bounding_box_vertices.push_back(points[i]);
        dashing_values.push_back(0);
        bounding_box_vertices.push_back(points[(i+1) % 6]);
        dashing_values.push_back(0);
        bounding_box_vertices.push_back(points[6+i]);
        dashing_values.push_back(0);
        bounding_box_vertices.push_back(points[6+(i+1) % 6]);
        dashing_values.push_back(0);
        bounding_box_vertices.push_back(points[i]);
        dashing_values.push_back(0);
        bounding_box_vertices.push_back(points[6+i]);
        dashing_values.push_back(0);

        if (periodic_boundary_condition_lengths.x > 0) {
            bounding_box_vertices.push_back(points[i]);
            dashing_values.push_back(0);
            bounding_box_vertices.push_back(center+v3+glm::normalize(points[i]-v3-center)*(radius+periodic_boundary_condition_lengths.x));
            dashing_values.push_back(dashes_per_length*periodic_boundary_condition_lengths.x*2);
            bounding_box_vertices.push_back(points[6+i]);
            dashing_values.push_back(0);
            bounding_box_vertices.push_back(center-v3+glm::normalize(points[6+i]+v3-center)*(radius+periodic_boundary_condition_lengths.x));
            dashing_values.push_back(dashes_per_length*periodic_boundary_condition_lengths.x*2);
        }

        if (periodic_boundary_condition_lengths.y > 0) {
            bounding_box_vertices.push_back(points[i]);
            dashing_values.push_back(0);
            bounding_box_vertices.push_back(points[i]+glm::normalize(v3)*periodic_boundary_condition_lengths.y);
            dashing_values.push_back(dashes_per_length*periodic_boundary_condition_lengths.y*2);
            bounding_box_vertices.push_back(points[6+i]);
            dashing_values.push_back(0);
            bounding_box_vertices.push_back(points[6+i]-glm::normalize(v3)*periodic_boundary_condition_lengths.y);
            dashing_values.push_back(dashes_per_length*periodic_boundary_condition_lengths.y*2);
        }
    }
    return BoundingBoxRenderer(view, bounding_box_vertices, dashing_values);
}
}
