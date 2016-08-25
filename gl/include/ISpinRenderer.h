#ifndef ISPINRENDERER_H
#define ISPINRENDERER_H
#include <vector>
#include <array>
#include <string>
#include "glm/glm.hpp"
#include "options.h"

class GLSpins;

class ISpinRenderer {
public:
  enum Option {
    VERTICAL_FIELD_OF_VIEW=100,
    BACKGROUND_COLOR,
    COLORMAP_IMPLEMENTATION,
    Z_RANGE,
    CAMERA_POSITION,
    CENTER_POSITION,
    UP_VECTOR
  };
  
  virtual ~ISpinRenderer() {};
  virtual void updateOptions(const Options<GLSpins>& options);
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions) = 0;
  virtual void draw(float aspect_ratio) const = 0;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions) = 0;
  Options<GLSpins> _options;
};

inline void ISpinRenderer::updateOptions(const Options<GLSpins>& options) {
  auto updatedOptions = _options.update(options);
  if (updatedOptions.size() > 0) {
    optionsHaveChanged(updatedOptions);
  }
}

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW> {
  float default_value = 45.0;
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::BACKGROUND_COLOR> {
 glm::vec3 default_value = {0.0, 0.0, 0.0};
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION> {
  std::string default_value = "\
  vec3 colormap(vec3 direction) {\
    return vec3(1.0, 1.0, 1.0);\
  }\
  ";
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::Z_RANGE> {
  glm::vec2 default_value = {-1, 1};
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::CAMERA_POSITION> {
  glm::vec3 default_value = {14.5, 14.5, 30};
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::CENTER_POSITION> {
  glm::vec3 default_value = {14.5, 14.5, 0};
};

template<> template<>
struct Options<GLSpins>::Option<ISpinRenderer::Option::UP_VECTOR> {
  glm::vec3 default_value = {0, 1, 0};
};

#endif
