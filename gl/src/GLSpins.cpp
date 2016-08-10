
#ifndef __gl_h_
#include <glad/glad.h>
#endif

#include <iostream>
#include <cmath>
#include <memory>

// Include GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtx/string_cast.hpp"
#include <glm/gtx/transform.hpp>

#include "Camera.h"
#include "GLSpins.h"
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

GLSpins::GLSpins() {
  if (!gladLoadGL()) {
    std::cerr << "Failed to initialize glad" << std::endl;
  }
  
  // Reset any error potentially caused by glad
  glGetError();
  
  CHECK_GL_ERROR;
  setCameraToDefault();

  glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  
  Options<GLSpins> options;
  options.set<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>(getColormapImplementation("hsv"));
  options.set<SurfaceSpinRendererOptions::SURFACE_INDICES>(SurfaceSpinRenderer::generateCartesianSurfaceIndices(30, 30));
  updateOptions(options);
  
  updateRenderers();
  CHECK_GL_ERROR;
}

GLSpins::~GLSpins() {
}

void GLSpins::updateSpins(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& directions) {
  CHECK_GL_ERROR;
	for (auto it : _renderers) {
    auto renderer = it.first;
    renderer->updateSpins(positions, directions);
  }
  CHECK_GL_ERROR;
}

void GLSpins::updateSystemGeometry(glm::vec3 bounds_min, glm::vec3 center, glm::vec3 bounds_max) {
  Options<GLSpins> options;
  options.set<GLSpins::Option::BOUNDING_BOX_MIN>(bounds_min);
  options.set<GLSpins::Option::BOUNDING_BOX_MAX>(bounds_max);
  options.set<GLSpins::Option::SYSTEM_CENTER>(center);
  updateOptions(options);
}

void GLSpins::draw() {
  CHECK_GL_ERROR;
  // Clear the screen and the depth buffer
  auto background_color = _options.get<ISpinRenderer::Option::BACKGROUND_COLOR>();
  glClearColor(background_color.x, background_color.y, background_color.z, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  Options<GLSpins> options;
  options.set<ISpinRenderer::Option::CAMERA_POSITION>(_camera.cameraPosition());
  options.set<ISpinRenderer::Option::CENTER_POSITION>(_camera.centerPosition());
  options.set<ISpinRenderer::Option::UP_VECTOR>(_camera.upVector());
  auto bounds_min = _options.get<GLSpins::Option::BOUNDING_BOX_MIN>();
  auto bounds_max = _options.get<GLSpins::Option::BOUNDING_BOX_MAX>();
  auto center = _options.get<GLSpins::Option::SYSTEM_CENTER>();
  options.set<CoordinateSystemRendererOptions::ORIGIN>(center);
  options.set<CoordinateSystemRendererOptions::AXIS_LENGTH>({
    fmax(fabs(bounds_max[0]-center[0]), 1.0),
    fmax(fabs(bounds_max[1]-center[1]), 1.0),
    fmax(fabs(bounds_max[2]-center[2]), 1.0)
  });
  updateOptions(options);
  
  glClear(GL_COLOR_BUFFER_BIT);
  for (auto it : _renderers) {
    auto renderer = it.first;
    auto viewport = it.second;
    glViewport((GLint)(viewport[0] * _width), (GLint)(viewport[1] * _height), (GLsizei)(viewport[2] * _width), (GLsizei)(viewport[3] * _height));
    glClear(GL_DEPTH_BUFFER_BIT);
    renderer->draw(viewport[2]/viewport[3] * _camera.aspectRatio());
  }
  _fps_counter.tick();
  CHECK_GL_ERROR;
}

float GLSpins::getFramerate() const {
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
      float angle = -length * 0.1f / 180 * 3.14f;
      auto rotation_matrix = glm::rotate(angle, axis);
      up = glm::mat3(rotation_matrix) * up;
      forward = glm::mat3(rotation_matrix) * forward;
      _camera.lookAt(_camera.centerPosition() - forward * camera_distance,
                     _camera.centerPosition(),
                     up);
    }
      break;
    case GLSpins::CameraMovementModes::TRANSLATE: {
      float factor = 0.001f * camera_distance * length;
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

void GLSpins::mouseScroll(const float& wheel_delta) {
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

void GLSpins::setFramebufferSize(float width, float height) {
  _width = width;
  _height = height;
  _camera.setAspectRatio(width / height);
}

void GLSpins::setCameraToDefault() {
  auto center = _options.get<GLSpins::Option::SYSTEM_CENTER>();
  _camera.lookAt(glm::vec3(center.x, center.y, center.z+30.0),
                 center,
                 glm::vec3(0.0f, 1.0f, 0.0f));
}

void GLSpins::setCameraToX() {
  auto center = _options.get<GLSpins::Option::SYSTEM_CENTER>();
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x+camera_distance, center.y, center.z),
                 center,
                 glm::vec3(0.0, 0.0, 1.0));
}

void GLSpins::setCameraToY() {
  auto center = _options.get<GLSpins::Option::SYSTEM_CENTER>();
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x, center.y+camera_distance, center.z),
                 center,
                 glm::vec3(1.0, 0.0, 0.0));
}

void GLSpins::setCameraToZ() {
  auto center = _options.get<GLSpins::Option::SYSTEM_CENTER>();
  float camera_distance = glm::length(_camera.centerPosition() - _camera.cameraPosition());
  _camera.lookAt(glm::vec3(center.x, center.y, center.z+camera_distance),
                 center,
                 glm::vec3(0.0, 1.0, 0.0));
}

static std::array<float, 4> locationToViewport(GLSpins::WidgetLocation location) {
  switch(location) {
    case GLSpins::WidgetLocation::BOTTOM_LEFT:
      return {0.0f, 0.0f, 0.2f, 0.2f};
    case GLSpins::WidgetLocation::BOTTOM_RIGHT:
      return {0.8f, 0.0f, 0.2f, 0.2f};
    case GLSpins::WidgetLocation::TOP_LEFT:
      return {0.0f, 0.8f, 0.2f, 0.2f};
    case GLSpins::WidgetLocation::TOP_RIGHT:
      return {0.8f, 0.8f, 0.2f, 0.2f};
  }
  return {0, 0, 1, 1};
}

void GLSpins::updateRenderers() {
  CHECK_GL_ERROR;
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
    renderer->updateOptions(options());
  }
  CHECK_GL_ERROR;
}

void GLSpins::updateOptions(const Options<GLSpins>& options) {
  CHECK_GL_ERROR;
  auto changedOptions = _options.update(options);
  if (changedOptions.size() == 0) {
    return;
  }
  optionsHaveChanged(changedOptions);
  for (auto it : _renderers) {
    auto renderer = it.first;
    renderer->updateOptions(options);
  }
  CHECK_GL_ERROR;
}

void GLSpins::options(const Options<GLSpins>& options) {
  _options = options;
  updateOptions(Options<GLSpins>());
}

const Options<GLSpins>& GLSpins::options() const {
  return _options;
}

void GLSpins::optionsHaveChanged(const std::vector<int>& changedOptions) {
  CHECK_GL_ERROR;
  bool renderersChanged = false;
  for (int option_id : changedOptions) {
    switch (option_id) {
      case GLSpins::Option::SHOW_MINIVIEW:
      case GLSpins::Option::SHOW_COORDINATE_SYSTEM:
      case GLSpins::Option::SHOW_BOUNDING_BOX:
      case GLSpins::Option::MINIVIEW_LOCATION:
      case GLSpins::Option::COORDINATE_SYSTEM_LOCATION:
      case GLSpins::Option::VISUALIZATION_MODE:
        renderersChanged = true;
        break;
    }
  }
  if (renderersChanged) {
    updateRenderers();
  }
  CHECK_GL_ERROR;
}
