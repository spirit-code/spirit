#ifndef VFRENDERING_RENDERER_BASE_HXX
#define VFRENDERING_RENDERER_BASE_HXX

#include <vector>

#include <glm/glm.hpp>

#include <VFRendering/Options.hxx>
#include <VFRendering/View.hxx>

namespace VFRendering {
class RendererBase {
public:

    RendererBase(const View& view);

    virtual ~RendererBase() {};
    virtual void update(bool keep_geometry) = 0;
    virtual void draw(float aspect_ratio) = 0;
    virtual void options(const Options& options);
    virtual void updateOptions(const Options& options);
    template<int index>
    void setOption(const typename Options::Type<index>::type& value);
    virtual void optionsHaveChanged(const std::vector<int>& changed_options);

protected:
    const Options& options() const;
    const std::vector<glm::vec3>& positions() const;
    const std::vector<glm::vec3>& directions() const;
    const std::vector<std::array<Geometry::index_type, 3>>& surfaceIndices() const;
    const std::vector<std::array<Geometry::index_type, 4>>& volumeIndices() const;

private:
    const View& m_view;
    Options m_options;
};

template<int index>
void RendererBase::setOption(const typename Options::Type<index>::type& value) {
    updateOptions(Options::withOption<index>(value));
}

}

#endif
