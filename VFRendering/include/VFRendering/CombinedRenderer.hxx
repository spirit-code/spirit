#ifndef VFRENDERING_COMBINED_RENDERER_HXX
#define VFRENDERING_COMBINED_RENDERER_HXX

#include <vector>
#include <memory>

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class CombinedRenderer : public RendererBase {
public:
    CombinedRenderer(const View& view, const std::vector<std::shared_ptr<RendererBase>>& renderers);
    virtual ~CombinedRenderer();
    virtual void updateIfNecessary() override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
    virtual void updateOptions(const Options& options) override;

protected:
    virtual void update(bool keep_geometry) override;

private:
    std::vector<std::shared_ptr<RendererBase>> m_renderers;
};
}

#endif
