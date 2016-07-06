#include "Camera.h"


void Camera::translate(const glm::vec3 &dt) {
    glm::mat4 translation_matrix(glm::translate(glm::mat4(1.0f), dt));
    m_camera = translation_matrix * m_camera;
}

void Camera::rotate(const glm::mat4 &dr) {
    // m_camera = dr * m_camera;
}

void Camera::rotate(float angle, const glm::vec3 &axis) {
    m_camera = glm::translate(glm::rotate(glm::translate(glm::mat4(1.0f), -m_camera_focus), angle, axis), m_camera_focus) * m_camera;
}

float Camera::z_func(float x, float y) {
    float r_quad = x*x + y*y;
    if(r_quad <= m_radius*m_radius / 2.0f) {
        return sqrt(m_radius*m_radius - r_quad);
    }
    else {
        return m_radius*m_radius / (2*sqrt(r_quad));
    }
}

void Camera::startTrackball(int px, int py) {
    m_previous_normalized_position_2d = glm::vec2(m_screen_to_normalized_2d * glm::vec3(px, py, 1.0f));
}

void Camera::updateTrackball(int px, int py) {
    float x1, y1, z1;
    float x2, y2, z2;
    glm::vec2 current_normalized_pos_2d(m_screen_to_normalized_2d * glm::vec3(px, py, 1.0f));

    if(glm::length(current_normalized_pos_2d - m_previous_normalized_position_2d) < 10e-8) {
        return;
    }

    x1 = m_previous_normalized_position_2d[0];
    y1 = m_previous_normalized_position_2d[1];
    z1 = z_func(x1, y1);
    x2 = current_normalized_pos_2d[0];
    y2 = current_normalized_pos_2d[1];
    z2 = z_func(x2, y2);

    glm::vec3 v1 = glm::normalize(glm::vec3(x1, y1, z1));
    glm::vec3 v2 = glm::normalize(glm::vec3(x2, y2, z2));
    glm::vec3 axis = glm::normalize(glm::cross(v1, v2));
    float angle = acosf(glm::dot(v1, v2));

    m_previous_normalized_position_2d = current_normalized_pos_2d;

    rotate(angle, axis);
}

void Camera::setWindowSize(int window_width, int window_height) {
    m_screen_to_normalized_2d = glm::mat3(2.0f / window_width, 0.0f, 0.0f,
                                          0.0f, -2.0f / window_height, 0.0f,
                                          -1.0f, 1.0f, 1.0f);
}
