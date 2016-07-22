#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include <glm/glm.hpp>

class Camera {
public:
  Camera(const glm::vec3& camera_position,
         const glm::vec3& center_position,
         const glm::vec3& up_vector,
         const double& aspect_ratio);

  void setAspectRatio(double aspect_ratio);
  
  void lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up);
  
  const glm::vec3& cameraPosition() const;
  const glm::vec3& centerPosition() const;
  const glm::vec3& upVector() const;
  const double& aspectRatio() const;
  
private:
  glm::vec3 _camera_position;
  glm::vec3 _center_position;
  glm::vec3 _up_vector;
  double _aspect_ratio;
};

#endif // CAMERA_H
