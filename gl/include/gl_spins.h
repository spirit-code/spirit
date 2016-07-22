#ifndef GL_SPINS_H
#define GL_SPINS_H

#include <array>
#include <chrono>
#include <memory>
#include <queue>

#ifndef __gl_h_
#include <glad/glad.h>
#endif

#include "glm/glm.hpp"
#include "Camera.h"
#include "data/Spin_System.h"
#include "data/Geometry.h"

class ISpinRenderer;

class FPSCounter {
public:
  void tick();
  double getFramerate() const;
private:
  int _max_n = 60;
  std::chrono::duration<double> _n_frame_duration = std::chrono::duration<double>::zero();
  std::chrono::steady_clock::time_point _previous_frame_time_point;
  std::queue<std::chrono::duration<double>> _frame_durations;
};


class GLSpins
{
public:
  enum CameraMovementModes {
    TRANSLATE,
    ROTATE
  };
  
  GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height);
  ~GLSpins();
  void draw();
  void update_spin_system(std::shared_ptr<Data::Spin_System> s);

  void mouseMove(const glm::vec2& position_before, const glm::vec2& position_after, CameraMovementModes mode);
  void mouseScroll(const double& wheel_delta);
  void setFramebufferSize(double width, double height);
  double getFramerate() const;

private:
  std::shared_ptr<Data::Spin_System> s;
  std::vector<std::pair<std::shared_ptr<ISpinRenderer>, std::array<double, 4>>> renderers;
  GLuint nos;
  glm::vec3 center;
  glm::vec3 bounds_min;
  glm::vec3 bounds_max;
  Camera _camera;
  FPSCounter _fps_counter;
  double _width;
  double _height;
  
  // Light color
  GLfloat light_color[3] = {1.0f, 1.0f, 1.0f};
  // Light direction given in camera coordinates (coordinate system after applying model and view matrix)
  GLfloat light_direction_cameraspace[3] = {-0.57735027f, 0.57735027f, 0.57735027f};

};

#endif
