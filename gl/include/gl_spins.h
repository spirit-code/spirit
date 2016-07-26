#ifndef GL_SPINS_H
#define GL_SPINS_H

#include <array>
#include <memory>

#ifndef __gl_h_
#include <glad/glad.h>
#endif

#include "glm/glm.hpp"
#include "Camera.h"
#include "data/Spin_System.h"
#include "data/Geometry.h"
#include "options.h"
#include "utilities.h"

class ISpinRenderer;

class GLSpins
{
public:
  enum CameraMovementModes {
    TRANSLATE,
    ROTATE
  };
  enum WidgetLocation {
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    TOP_LEFT,
    TOP_RIGHT
  };
  enum VisualizationMode {
    SURFACE,
    ARROWS,
    SPHERE
  };
  enum Option {
    SHOW_BOUNDING_BOX,
    SHOW_MINIVIEW,
    MINIVIEW_LOCATION,
    SHOW_COORDINATE_SYSTEM,
    COORDINATE_SYSTEM_LOCATION,
    VISUALIZATION_MODE
  };
  
  GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height);
  ~GLSpins();
  void draw();
  void update_spin_system(std::shared_ptr<Data::Spin_System> s);

  void mouseMove(const glm::vec2& position_before, const glm::vec2& position_after, CameraMovementModes mode);
  void mouseScroll(const double& wheel_delta);
  void setCameraToDefault();
  void setCameraToX();
  void setCameraToY();
  void setCameraToZ();
  void setFramebufferSize(double width, double height);
  double getFramerate() const;
  
  void updateOptions(const Options<GLSpins>& options);
  void options(const Options<GLSpins>& options);
  const Options<GLSpins>& options() const;
  
private:
  
  void updateRenderers();
  void optionsHaveChanged(const std::vector<int>& changedOptions);
  
  std::shared_ptr<Data::Spin_System> s;
  glm::vec3 center;
  glm::vec3 bounds_min;
  glm::vec3 bounds_max;
  std::vector<std::pair<std::shared_ptr<ISpinRenderer>, std::array<double, 4>>> _renderers;
  Camera _camera;
  FPSCounter _fps_counter;
  double _width;
  double _height;
  
  Options<GLSpins> _options;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::SHOW_BOUNDING_BOX> {
  bool default_value = false;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::SHOW_MINIVIEW> {
  bool default_value = false;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::MINIVIEW_LOCATION> {
  GLSpins::WidgetLocation default_value = GLSpins::WidgetLocation::BOTTOM_LEFT;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::SHOW_COORDINATE_SYSTEM> {
  bool default_value = false;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::COORDINATE_SYSTEM_LOCATION> {
  GLSpins::WidgetLocation default_value = GLSpins::WidgetLocation::BOTTOM_RIGHT;
};

template<> template<>
struct Options<GLSpins>::Option<GLSpins::Option::VISUALIZATION_MODE> {
  GLSpins::VisualizationMode default_value = GLSpins::VisualizationMode::ARROWS;
};

#endif
