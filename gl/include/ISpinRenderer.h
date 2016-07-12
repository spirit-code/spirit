#ifndef ISPINRENDERER_H
#define ISPINRENDERER_H
#include <vector>
#include "glm/glm.hpp"

class ISpinRenderer {
public:
  virtual ~ISpinRenderer() {};
  virtual void updateOptions() = 0;
  virtual void initGL() = 0;
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions) = 0;
  virtual void draw(double aspect_ratio) const = 0;
};

#endif
