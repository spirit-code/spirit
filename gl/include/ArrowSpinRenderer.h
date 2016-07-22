#ifndef ARROWSPINRENDERER_H
#define ARROWSPINRENDERER_H
#include "ISpinRenderer.h"

class ArrowSpinRenderer : public ISpinRenderer {
public:
  ArrowSpinRenderer();
  virtual ~ArrowSpinRenderer();
  virtual void initGL();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(double aspectRatio) const;
  
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
  CONE_RADIUS = 100,
  CONE_HEIGHT,
  CYLINDER_RADIUS,
  CYLINDER_HEIGHT,
  LEVEL_OF_DETAIL
};

template<> template<>
struct Options<ISpinRenderer>::Option<ArrowSpinRendererOptions::CONE_RADIUS> {
  double default_value = 0.25;
};

template<> template<>
struct Options<ISpinRenderer>::Option<ArrowSpinRendererOptions::CONE_HEIGHT> {
  double default_value = 0.6;
};

template<> template<>
struct Options<ISpinRenderer>::Option<ArrowSpinRendererOptions::CYLINDER_RADIUS> {
  double default_value = 0.125;
};

template<> template<>
struct Options<ISpinRenderer>::Option<ArrowSpinRendererOptions::CYLINDER_HEIGHT> {
  double default_value = 0.7;
};

template<> template<>
struct Options<ISpinRenderer>::Option<ArrowSpinRendererOptions::LEVEL_OF_DETAIL> {
  unsigned int default_value = 20;
};

#endif
