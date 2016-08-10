#ifndef BOUNDINGBOXRENDERER_H
#define BOUNDINGBOXRENDERER_H
#include <array>
#include "ISpinRenderer.h"

class BoundingBoxRenderer : public ISpinRenderer {
public:
  enum Option {
    COLOR = 600
  };
  
  BoundingBoxRenderer();
  virtual ~BoundingBoxRenderer();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(float aspectRatio) const;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);

private:
  void _updateVertexData();

  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _vbo = 0;
};

template<> template<>
struct Options<GLSpins>::Option<BoundingBoxRenderer::Option::COLOR> {
  glm::vec3 default_value = {1.0, 1.0, 1.0};
};

#endif
