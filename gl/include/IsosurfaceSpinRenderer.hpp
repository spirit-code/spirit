#ifndef ISOSURFACESPINRENDERER_H
#define ISOSURFACESPINRENDERER_H
#include "ISpinRenderer.hpp"

class IsosurfaceSpinRenderer : public ISpinRenderer {
public:
  IsosurfaceSpinRenderer();
  virtual ~IsosurfaceSpinRenderer();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(float aspectRatio) const;
  
protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);
  
private:
  void _updateShaderProgram();
  void _updateIsosurfaceIndices();
  
  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _ibo = 0;
  GLuint _instancePositionVbo = 0;
  GLuint _instanceDirectionVbo = 0;
  unsigned int _numIndices = 0;
  std::vector<glm::vec3> _positions;
  std::vector<glm::vec3> _directions;
};

enum IsosurfaceSpinRendererOptions {
  ISOVALUE = 700
};

template<> template<>
struct Options<GLSpins>::Option<IsosurfaceSpinRendererOptions::ISOVALUE> {
  double default_value = 0;
};

#endif
