#include "CombinedSpinRenderer.hpp"


CombinedSpinRenderer::CombinedSpinRenderer(const std::vector<std::shared_ptr<ISpinRenderer>>& renderers) : _renderers(renderers) {
}

CombinedSpinRenderer::~CombinedSpinRenderer() {
}

void CombinedSpinRenderer::updateOptions(const Options<GLSpins>& options) {
  for (auto renderer : _renderers) {
    renderer->updateOptions(options);
  }
}

void CombinedSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                       const std::vector<glm::vec3>& directions) {
  for (auto renderer : _renderers) {
    renderer->updateSpins(positions, directions);
  }
}

void CombinedSpinRenderer::draw(float aspectRatio) const {
  for (auto renderer : _renderers) {
    renderer->draw(aspectRatio);
  }
}

void CombinedSpinRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
}
