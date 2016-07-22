#ifndef ISPINRENDERER_H
#define ISPINRENDERER_H
#include <vector>
#include <array>
#include "glm/glm.hpp"
#include "options.h"

class ISpinRenderer {
public:
  virtual ~ISpinRenderer() {};
  void updateOptions(const Options<ISpinRenderer>& options);
  virtual void initGL() = 0;
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions) = 0;
  virtual void draw(double aspect_ratio) const = 0;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions) = 0;
  Options<ISpinRenderer> _options;
};

inline void ISpinRenderer::updateOptions(const Options<ISpinRenderer>& options) {
  auto updatedOptions = _options.update(options);
  if (updatedOptions.size() > 0) {
    optionsHaveChanged(updatedOptions);
  }
}

enum ISpinRendererOptions {
  VERTICAL_FIELD_OF_VIEW,
  BACKGROUND_COLOR,
  COLORMAP_IMPLEMENTATION,
  Z_RANGE,
  CAMERA_POSITION,
  CENTER_POSITION,
  UP_VECTOR
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::VERTICAL_FIELD_OF_VIEW> {
  double default_value = 45.0;
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::BACKGROUND_COLOR> {
 glm::vec3 default_value = {1.0, 1.0, 1.0};
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::COLORMAP_IMPLEMENTATION> {
  std::string default_value = "\
  vec3 colormap(vec3 direction) {\
    return vec3(1.0, 1.0, 1.0);\
  }\
  ";
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::Z_RANGE> {
  glm::vec2 default_value = {-10, 10};
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::CAMERA_POSITION> {
  glm::vec3 default_value = {14.5, 14.5, 30};
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::CENTER_POSITION> {
  glm::vec3 default_value = {14.5, 14.5, 0};
};

template<> template<>
struct Options<ISpinRenderer>::Option<ISpinRendererOptions::UP_VECTOR> {
  glm::vec3 default_value = {0, 1, 0};
};

#endif
