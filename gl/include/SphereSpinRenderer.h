#ifndef SPHERESPINRENDERER_H
#define SPHERESPINRENDERER_H
#include "ISpinRenderer.h"

class SphereSpinRenderer : public ISpinRenderer {
public:
  SphereSpinRenderer();
  virtual ~SphereSpinRenderer();
  virtual void initGL();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(double aspectRatio) const;
  
protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);
  
private:
  void _updateShaderProgram();
  
  GLuint _program1 = 0;
  GLuint _program2 = 0;
  GLuint _vao1 = 0;
  GLuint _vao2 = 0;
  GLuint _fakeSphereVbo = 0;
  GLuint _instanceDirectionVbo = 0;
  unsigned int _numInstances = 0;
};

enum SphereSpinRendererOptions {
  POINT_SIZE_RANGE = 400,
  INNER_SPHERE_RADIUS,
  USE_SPHERE_FAKE_PERSPECTIVE
};

template<> template<>
struct Options<GLSpins>::Option<SphereSpinRendererOptions::POINT_SIZE_RANGE> {
  glm::vec2 default_value = {1.0, 4.0};
};

template<> template<>
struct Options<GLSpins>::Option<SphereSpinRendererOptions::INNER_SPHERE_RADIUS> {
  double default_value = 0.95;
};

template<> template<>
struct Options<GLSpins>::Option<SphereSpinRendererOptions::USE_SPHERE_FAKE_PERSPECTIVE> {
  bool default_value = true;
};

#endif
