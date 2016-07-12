#ifndef ARROWSPINRENDERER_H
#define ARROWSPINRENDERER_H
#include "ISpinRenderer.h"

class ArrowSpinRenderer : public ISpinRenderer {
public:
  ArrowSpinRenderer();
  virtual ~ArrowSpinRenderer();
  virtual void updateOptions();
  virtual void initGL();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(double aspectRatio) const;

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

#endif
