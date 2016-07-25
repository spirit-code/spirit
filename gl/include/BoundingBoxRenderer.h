#ifndef BOUNDINGBOXRENDERER_H
#define BOUNDINGBOXRENDERER_H
#include <array>
#include "ISpinRenderer.h"

class BoundingBoxRenderer : public ISpinRenderer {
public:
  BoundingBoxRenderer();
  virtual ~BoundingBoxRenderer();
  virtual void initGL();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(double aspectRatio) const;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);

private:
  void _updateVertexData();

  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _vbo = 0;
};

enum BoundingBoxRendererOptions {
  COLOR = 400,
  POSITION
};

template<> template<>
struct Options<ISpinRenderer>::Option<BoundingBoxRendererOptions::COLOR> {
  glm::vec3 default_value = {1.0, 1.0, 1.0};
};

template<> template<>
struct Options<ISpinRenderer>::Option<BoundingBoxRendererOptions::POSITION> {
  struct {
    glm::vec3 min;
    glm::vec3 max;
  } default_value = {{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};
};

#endif
