#include <iostream>
#include <cmath>
#include <cassert>
#include <memory>

// Include GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>

#include "Camera.h"
#include "Geometry.h"
#include "gl_spins.h"
#include "ISpinRenderer.h"
#include "ArrowSpinRenderer.h"
#include "SurfaceSpinRenderer.h"
#include "SphereSpinRenderer.h"
#include "utilities.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif // !M_PI


void FPSCounter::tick() {
  if (_previous_frame_time_point != std::chrono::steady_clock::time_point()) {
    auto previous_duration = std::chrono::steady_clock::now() - _previous_frame_time_point;
    _n_frame_duration += previous_duration;
    _frame_durations.push(previous_duration);
    while (_frame_durations.size() > _max_n) {
      _n_frame_duration -= _frame_durations.front();
      _frame_durations.pop();
    }
  }
  _previous_frame_time_point = std::chrono::steady_clock::now();
}

double FPSCounter::getFramerate() const {
  return _frame_durations.size() / _n_frame_duration.count();
}

double GLSpins::getFramerate() const {
  return _fps_counter.getFramerate();
}

void GLSpins::mouseMove(const glm::vec2& position_before, const glm::vec2& position_after, GLSpins::CameraMovementModes mode) {
  if (position_before == position_after) {
    return;
  }
  auto delta = position_after-position_before;
  auto length = glm::length(delta);
  auto forward = glm::normalize(_camera.centerPosition() - _camera.cameraPosition());
  auto camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  auto up = glm::normalize(_camera.upVector());
  auto right = glm::normalize(glm::cross(forward, up));
  up = glm::normalize(glm::cross(right, forward));
  delta = glm::normalize(delta);
  switch (mode) {
    case GLSpins::CameraMovementModes::ROTATE: {
      auto axis = glm::normalize(delta.x * up + delta.y * right);
      float angle = -length * 0.1 / 180 * 3.14;
      auto rotation_matrix = glm::rotate(angle, axis);
      up = glm::mat3(rotation_matrix) * up;
      forward = glm::mat3(rotation_matrix) * forward;
      _camera.lookAt(_camera.centerPosition() - forward * camera_distance,
                     _camera.centerPosition(),
                     up);
    }
      break;
    case GLSpins::CameraMovementModes::TRANSLATE: {
      float factor = 0.001 * camera_distance * length;
      auto translation = factor * (delta.y * up - delta.x * right);
      _camera.lookAt(_camera.cameraPosition() + translation,
                     _camera.centerPosition() + translation,
                     up);
    }
      break;
    default:
      break;
  }
}

void GLSpins::mouseScroll(const double& wheel_delta) {
  auto forward = _camera.centerPosition() - _camera.cameraPosition();
  float camera_distance = glm::length(forward);
  if (camera_distance < 2 && wheel_delta < 1) {
    return;
  }
  
  auto camera_position = _camera.centerPosition() - (float)(1+0.02*wheel_delta) * forward;
  _camera.lookAt(camera_position,
                 _camera.centerPosition(),
                 _camera.upVector());
}

void GLSpins::setFramebufferSize(double width, double height) {
  _width = width;
  _height = height;
  _camera.setAspectRatio(width / height);
}

Camera::Camera(const glm::vec3& camera_position,
               const glm::vec3& center_position,
               const glm::vec3& up_vector,
               const double& aspect_ratio) :
_camera_position(camera_position),
_center_position(center_position),
_up_vector(up_vector),
_aspect_ratio(aspect_ratio) {
}



GLSpins::GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height) :
_camera(glm::vec3(0, 0, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 1.0)
{
	// Copy Positions from geometry
	this->s = s;
	std::shared_ptr<Data::Geometry> g = s->geometry;
	this->nos = g->nos;

	// Copy Center and bounds
	center = glm::vec3(g->center[0], g->center[1], g->center[2]);
	bounds_min = glm::vec3(g->bounds_min[0], g->bounds_min[1], g->bounds_min[2]);
	bounds_max = glm::vec3(g->bounds_max[0], g->bounds_max[1], g->bounds_max[2]);

	this->_camera.lookAt(glm::vec3(center.x, center.y, 30.0f),
						center,
						glm::vec3(0.0f, 1.0f, 0.0f));

    // Initialize glad
    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize glad" << std::endl;
        // return 1;
    }
  
  
  // Spin positions
  std::vector<glm::vec3> translations(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    translations[i] = glm::vec3(g->spin_pos[0][i], g->spin_pos[1][i], g->spin_pos[2][i]);
  }
  
  // Spin orientations
  std::vector<glm::vec3> directions(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    directions[i] = glm::vec3(s->spins[i], s->spins[g->nos + i], s->spins[2*g->nos + i]);
  }
  
  std::array<double, 4> viewport = {0, 0, 1, 1};
  std::shared_ptr<ISpinRenderer> renderer = std::make_shared<SphereSpinRenderer>();
  Options<ISpinRenderer> options;
  options.set<ISpinRendererOptions::COLORMAP_IMPLEMENTATION>(getColormapImplementation("hsv"));
  options.set<SurfaceSpinRendererOptions::SURFACE_INDICES>(SurfaceSpinRenderer::generateCartesianSurfaceIndices(30, 30));
  renderer->initGL();
  renderer->updateOptions(options);
  renderer->updateSpins(translations, directions);
  renderers.push_back({renderer, viewport});
  std::shared_ptr<ISpinRenderer> renderer2 = std::make_shared<ArrowSpinRenderer>();
  renderer2->initGL();
  renderer2->updateOptions(options);
  renderer2->updateSpins(translations, directions);
  renderers.push_back({renderer2, {0, 0, 0.2, 0.2}});

    // Dark blue background
    //glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	// Dark gray background
	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
}

void GLSpins::draw() {
  // Clear the screen and the depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  std::shared_ptr<Data::Geometry> g = s->geometry;
  // Spin positions
  std::vector<glm::vec3> translations(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    translations[i] = glm::vec3(g->spin_pos[0][i], g->spin_pos[1][i], g->spin_pos[2][i]);
  }
  
  // Spin orientations
  std::vector<glm::vec3> directions(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    directions[i] = glm::vec3(s->spins[i], s->spins[g->nos + i], s->spins[2*g->nos + i]);
  }
  
  Options<ISpinRenderer> options;
  options.set<ISpinRendererOptions::CAMERA_POSITION>(_camera.cameraPosition());
  options.set<ISpinRendererOptions::CENTER_POSITION>(_camera.centerPosition());
  options.set<ISpinRendererOptions::UP_VECTOR>(_camera.upVector());
  for (auto it = renderers.begin(); it != renderers.end(); it++) {
    auto renderer = it->first;
    auto viewport = it->second;
    // TODO: glViewport
    glViewport(viewport[0] * _width, viewport[1] * _height, viewport[2] * _width, viewport[3] * _height);
    renderer->updateOptions(options);
    renderer->updateSpins(translations, directions);
    glClear(GL_DEPTH_BUFFER_BIT);
    renderer->draw(viewport[2]/viewport[3] * _camera.aspectRatio());
    assert(!glGetError());
  }
  _fps_counter.tick();
  assert(!glGetError());
}

void GLSpins::update_spin_system(std::shared_ptr<Data::Spin_System> s)
{
	this->s = s;
}

GLSpins::~GLSpins() {
}
