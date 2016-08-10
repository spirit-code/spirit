#ifndef SURFACESPINRENDERER_H
#define SURFACESPINRENDERER_H
#include "ISpinRenderer.h"

class SurfaceSpinRenderer : public ISpinRenderer {
public:
  SurfaceSpinRenderer();
  virtual ~SurfaceSpinRenderer();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(float aspectRatio) const;
  
  static std::vector<unsigned int> generateCartesianSurfaceIndices(int nx, int ny);
  
protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);
  
private:
  void _updateShaderProgram();
  void _updateSurfaceIndices();
  
  GLuint _program = 0;
  GLuint _vao = 0;
  GLuint _ibo = 0;
  GLuint _instancePositionVbo = 0;
  GLuint _instanceDirectionVbo = 0;
  unsigned int _numIndices = 0;
};

enum SurfaceSpinRendererOptions {
  SURFACE_INDICES = 300
};

template<> template<>
struct Options<GLSpins>::Option<SurfaceSpinRendererOptions::SURFACE_INDICES> {
  std::vector<unsigned int> default_value;
};

#endif
