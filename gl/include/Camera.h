#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>


class Camera
{
    public:
        // constructors
        Camera();
        Camera(glm::vec3 pos, glm::vec3 forward, glm::vec3 up, float trackball_radius);
        Camera(float pos_x, float pos_y, float pos_z,
               float forward_x, float forward_y, float forward_z,
               float up_x, float up_y, float up_z,
               float trackball_radius);

        // transform by (translate / rotate)
        void translate(const glm::vec3 &dt);
        void translate(float dx, float dy, float dz);
        void translate(int dx, int dy);
        void translate(int dz);
        void rotate(const glm::mat4 &dr);
        void rotate(const glm::mat3 &dr);
        void rotate(float angle, const glm::vec3 &axis);
        void rotate(float angle, float ax, float ay, float az);

        // trackball control
        void startTrackball(int px, int py);
        void updateTrackball(int px, int py);
        void setWindowSize(int window_width, int window_height);

		// setters
		void lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up);

        // accessors
        const glm::mat4 &view_matrix() const;

    private:
        float z_func(float x, float y);

        glm::vec2 m_previous_normalized_position_2d;
        glm::mat3 m_screen_to_normalized_2d;
        glm::mat4 m_camera;
        float m_radius;
        glm::vec3 m_camera_focus;   // camera focus point in camera space
};

// constructors

inline Camera::Camera() : m_camera(glm::lookAt(glm::vec3(00.0f, 00.0f, 20.0f),
                                               glm::vec3(0.0f, 0.0f, 0.0f),
                                               glm::vec3(0.0f, 1.0f, 0.0f))),
                          m_radius(0.5f),
                          m_camera_focus(-m_camera[3]) // camera looks at world origin
{ }

inline Camera::Camera(float pos_x, float pos_y, float pos_z,
                      float focus_x, float focus_y, float focus_z,
                      float up_x, float up_y, float up_z,
                      float trackball_radius) :
    m_camera(glm::lookAt(glm::vec3(pos_x, pos_y, pos_z),
                         glm::vec3(focus_x, focus_y, focus_z),
                         glm::vec3(up_x, up_y, up_z))),
    m_radius(trackball_radius),
    m_camera_focus(m_camera[3]) {  }

inline Camera::Camera(glm::vec3 pos, glm::vec3 forward, glm::vec3 up, float trackball_radius) :
    m_camera(glm::lookAt(pos, forward, up)), m_radius(trackball_radius) {  }

// transform By (Add/Scale)
inline void Camera::translate(float dx, float dy, float dz) { translate(glm::vec3(dx, dy, dz)); }
inline void Camera::translate(int dx, int dy) { translate(m_screen_to_normalized_2d * glm::vec3(dx*3.0, dy*3.0, 0.0f)); }
inline void Camera::translate(int dz) { translate(glm::vec3(0.0f, 0.0f, 0.005f*dz)); m_camera_focus[2] -= 0.005f*dz; }
inline void Camera::rotate(const glm::mat3 &dr) { rotate(glm::mat4(dr)); }
inline void Camera::rotate(float angle, float ax, float ay, float az) { rotate(angle, glm::vec3(ax, ay, az)); }

// setters
inline void Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up)
{
	m_camera = glm::lookAt(eye, center, up);
	m_camera_focus = glm::vec3(-m_camera[3])-center;
}


// accessors
inline const glm::mat4 &Camera::view_matrix() const { return m_camera; }

inline glm::mat4 operator*(const Camera &camera, const glm::mat4 &matrix)
{
    return camera.view_matrix() * matrix;
}

inline glm::mat4 operator*(const glm::mat4 &matrix, const Camera &camera)
{
    return matrix * camera.view_matrix();
}

#endif // CAMERA_H
