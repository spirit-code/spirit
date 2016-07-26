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
#include "BoundingBoxRenderer.h"
#include "CombinedSpinRenderer.h"
#include "CoordinateSystemRenderer.h"
#include "utilities.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif // !M_PI


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

void GLSpins::setCameraToDefault() {
  _camera.lookAt(glm::vec3(center.x, center.y, center.z+30.0),
                 center,
                 glm::vec3(0.0f, 1.0f, 0.0f));
}

void GLSpins::setCameraToX() {
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x+camera_distance, center.y, center.z),
                 center,
                 glm::vec3(0.0, 0.0, 1.0));
}

void GLSpins::setCameraToY() {
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x, center.y+camera_distance, center.z),
                           center,
                           glm::vec3(1.0, 0.0, 0.0));
}

void GLSpins::setCameraToZ() {
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x, center.y, center.z+camera_distance),
                           center,
                           glm::vec3(0.0, 1.0, 0.0));
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

	this->_camera.lookAt(glm::vec3(center.x, center.y, center.z+30.0f),
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
  
  Options<GLSpins> options;
  options.set<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>(getColormapImplementation("hsv"));
  options.set<SurfaceSpinRendererOptions::SURFACE_INDICES>(SurfaceSpinRenderer::generateCartesianSurfaceIndices(30, 30));
  updateOptions(options);
  
  updateRenderers();

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
  
  Options<GLSpins> options;
  options.set<ISpinRenderer::Option::CAMERA_POSITION>(_camera.cameraPosition());
  options.set<ISpinRenderer::Option::CENTER_POSITION>(_camera.centerPosition());
  options.set<ISpinRenderer::Option::UP_VECTOR>(_camera.upVector());
  options.set<BoundingBoxRendererOptions::POSITION>({
    {g->bounds_min[0], g->bounds_min[1], g->bounds_min[2]},
    {g->bounds_max[0], g->bounds_max[1], g->bounds_max[2]}
  });
  options.set<CoordinateSystemRendererOptions::ORIGIN>({
    g->center[0], g->center[1], g->center[2]
  });
  options.set<CoordinateSystemRendererOptions::AXIS_LENGTH>({
    fmax(fabs(g->bounds_max[0]-g->center[0]), 1.0),
    fmax(fabs(g->bounds_max[1]-g->center[1]), 1.0),
    fmax(fabs(g->bounds_max[2]-g->center[2]), 1.0)
  });
  updateOptions(options);
  
  glClear(GL_COLOR_BUFFER_BIT);
  for (auto it : _renderers) {
    auto renderer = it.first;
    auto viewport = it.second;
    // TODO: glViewport
    glViewport(viewport[0] * _width, viewport[1] * _height, viewport[2] * _width, viewport[3] * _height);
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

static std::array<double, 4> locationToViewport(GLSpins::WidgetLocation location) {
  switch(location) {
    case GLSpins::WidgetLocation::BOTTOM_LEFT:
      return {0, 0, 0.2, 0.2};
    case GLSpins::WidgetLocation::BOTTOM_RIGHT:
      return {0.8, 0, 0.2, 0.2};
    case GLSpins::WidgetLocation::TOP_LEFT:
      return {0, 0.8, 0.2, 0.2};
    case GLSpins::WidgetLocation::TOP_RIGHT:
      return {0.8, 0.8, 0.2, 0.2};
  }
  return {0, 0, 1, 1};
}

void GLSpins::updateRenderers() {
  bool show_bounding_box = options().get<GLSpins::Option::SHOW_BOUNDING_BOX>();
  bool show_coordinate_system = options().get<GLSpins::Option::SHOW_COORDINATE_SYSTEM>();
  bool show_miniview = options().get<GLSpins::Option::SHOW_MINIVIEW>();
  GLSpins::VisualizationMode mode = options().get<GLSpins::Option::VISUALIZATION_MODE>();
  GLSpins::WidgetLocation coordinate_system_location = options().get<GLSpins::Option::COORDINATE_SYSTEM_LOCATION>();
  GLSpins::WidgetLocation miniview_location = options().get<GLSpins::Option::MINIVIEW_LOCATION>();
  _renderers.clear();
  std::shared_ptr<ISpinRenderer> main_renderer;
  switch (mode) {
    case VisualizationMode::ARROWS:
      main_renderer = std::make_shared<ArrowSpinRenderer>();
      break;
    case VisualizationMode::SURFACE:
      main_renderer = std::make_shared<SurfaceSpinRenderer>();
      break;
    case VisualizationMode::SPHERE:
      main_renderer = std::make_shared<SphereSpinRenderer>();
      break;
  }
  if (show_bounding_box && mode != VisualizationMode::SPHERE) {
    std::vector<std::shared_ptr<ISpinRenderer>> r = {
      main_renderer,
      std::make_shared<BoundingBoxRenderer>()
    };
    main_renderer = std::make_shared<CombinedSpinRenderer>(r);
  }
  _renderers.push_back({main_renderer, {0, 0, 1, 1}});
  
  if (show_coordinate_system) {
    std::shared_ptr<ISpinRenderer> renderer = std::make_shared<CoordinateSystemRenderer>();
    _renderers.push_back({renderer, locationToViewport(coordinate_system_location)});
  }
  if (show_miniview) {
    std::shared_ptr<ISpinRenderer> renderer;
    if (mode == VisualizationMode::SPHERE) {
      renderer = std::make_shared<SurfaceSpinRenderer>();
      if (show_bounding_box) {
        std::vector<std::shared_ptr<ISpinRenderer>> r = {
          renderer,
          std::make_shared<BoundingBoxRenderer>()
        };
        renderer = std::make_shared<CombinedSpinRenderer>(r);
      }
    } else {
      renderer = std::make_shared<SphereSpinRenderer>();
    }
    _renderers.push_back({renderer, locationToViewport(miniview_location)});
  }

  for (auto it : _renderers) {
    auto renderer = it.first;
    renderer->initGL();
    renderer->updateOptions(options());
  }
}

void GLSpins::updateOptions(const Options<GLSpins>& options) {
  auto changedOptions = _options.update(options);
  if (changedOptions.size() == 0) {
    return;
  }
  optionsHaveChanged(changedOptions);
  for (auto it : _renderers) {
    auto renderer = it.first;
    renderer->updateOptions(options);
  }
}

void GLSpins::options(const Options<GLSpins>& options) {
  _options = options;
  updateOptions(Options<GLSpins>());
}

const Options<GLSpins>& GLSpins::options() const {
  return _options;
}

void GLSpins::optionsHaveChanged(const std::vector<int>& changedOptions) {
  bool renderersChanged = false;
  for (int option_id : changedOptions) {
    
  }
  if (renderersChanged) {
    updateRenderers();
  }
}
