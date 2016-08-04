#ifndef COMBINEDSPINRENDERER_H
#define COMBINEDSPINRENDERER_H
#include <vector>
#include <memory>
#include "ISpinRenderer.h"

class CombinedSpinRenderer : public ISpinRenderer {
public:
  CombinedSpinRenderer(const std::vector<std::shared_ptr<ISpinRenderer>>& renderers);
  virtual ~CombinedSpinRenderer();
  virtual void updateOptions(const Options<GLSpins>& options);
  virtual void initGL();
  virtual void updateSpins(const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec3>& directions);
  virtual void draw(double aspectRatio) const;

protected:
  virtual void optionsHaveChanged(const std::vector<int>& changedOptions);

private:
  std::vector<std::shared_ptr<ISpinRenderer>> _renderers;
};

#endif
