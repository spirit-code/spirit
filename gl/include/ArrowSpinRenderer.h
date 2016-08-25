#ifndef ARROWSPINRENDERER_H
#define ARROWSPINRENDERER_H
#include "ISpinRenderer.h"
#include "BoundingBoxRenderer.h"

class ArrowSpinRenderer : public ISpinRenderer {
public:
  ArrowSpinRenderer();
  virtual ~ArrowSpinRenderer();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(float aspectRatio) const;
  
protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);

private:
  void _updateShaderProgram();
  void _updateVertexData();

  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _vbo = 0;
  GLuint _ibo = 0;
  GLuint _instancePositionVbo = 0;
  GLuint _instanceDirectionVbo = 0;
  unsigned int _numIndices = 0;
  unsigned int _numInstances = 0;
};

enum ArrowSpinRendererOptions {
  CONE_RADIUS = 200,
  CONE_HEIGHT,
  CYLINDER_RADIUS,
  CYLINDER_HEIGHT,
  LEVEL_OF_DETAIL
};

template<> template<>
struct Options<GLSpins>::Option<ArrowSpinRendererOptions::CONE_RADIUS> {
  float default_value = 0.25f;
};

template<> template<>
struct Options<GLSpins>::Option<ArrowSpinRendererOptions::CONE_HEIGHT> {
  float default_value = 0.6f;
};

template<> template<>
struct Options<GLSpins>::Option<ArrowSpinRendererOptions::CYLINDER_RADIUS> {
  float default_value = 0.125f;
};

template<> template<>
struct Options<GLSpins>::Option<ArrowSpinRendererOptions::CYLINDER_HEIGHT> {
  float default_value = 0.7f;
};

template<> template<>
struct Options<GLSpins>::Option<ArrowSpinRendererOptions::LEVEL_OF_DETAIL> {
  unsigned int default_value = 20;
};

#endif
