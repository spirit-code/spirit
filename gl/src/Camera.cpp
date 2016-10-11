#include "Camera.hpp"

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera() : Camera(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 1.0) {
}

Camera::Camera(const glm::vec3& camera_position,
               const glm::vec3& center_position,
               const glm::vec3& up_vector,
               const float& aspect_ratio) :
_camera_position(camera_position),
_center_position(center_position),
_up_vector(up_vector),
_aspect_ratio(aspect_ratio) {
}

void Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
  _camera_position = eye;
  _center_position = center;
  _up_vector = up;
}

void Camera::setAspectRatio(float aspect_ratio) {
  _aspect_ratio = aspect_ratio;
}

const glm::vec3& Camera::cameraPosition() const {
  return _camera_position;
}

const glm::vec3& Camera::centerPosition() const {
  return _center_position;
}

const glm::vec3& Camera::upVector() const {
  return _up_vector;
}

const float& Camera::aspectRatio() const {
  return _aspect_ratio;
}
