#include "Camera.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

void Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
  _camera_position = eye;
  _center_position = center;
  _up_vector = up;
}

void Camera::setAspectRatio(double aspect_ratio) {
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

const double& Camera::aspectRatio() const {
  return _aspect_ratio;
}
