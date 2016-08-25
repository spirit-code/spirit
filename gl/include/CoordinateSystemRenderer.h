#ifndef COORDINATESYSTEMRENDERER_H
#define COORDINATESYSTEMRENDERER_H
#include <array>
#include "ISpinRenderer.h"

class CoordinateSystemRenderer : public ISpinRenderer {
public:
  CoordinateSystemRenderer();
  virtual ~CoordinateSystemRenderer();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(float aspectRatio) const;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);

private:
  void _updateShaderProgram();

  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _vbo = 0;
};

enum CoordinateSystemRendererOptions {
  AXIS_LENGTH = 500,
  ORIGIN
};

template<> template<>
struct Options<GLSpins>::Option<CoordinateSystemRendererOptions::AXIS_LENGTH> {
  glm::vec3 default_value = {1.0, 1.0, 1.0};
};

template<> template<>
struct Options<GLSpins>::Option<CoordinateSystemRendererOptions::ORIGIN> {
  glm::vec3 default_value = {0.0, 0.0, 0.0};
};

#endif
