#ifndef VFRENDERING_SPHERE_RENDERER_HXX
#define VFRENDERING_SPHERE_RENDERER_HXX

#include <VFRendering/GlyphRenderer.hxx>

namespace VFRendering {
class SphereRenderer : public GlyphRenderer {
public:
    enum Option {
        SPHERE_RADIUS = 800,
        LEVEL_OF_DETAIL
    };

    SphereRenderer(const View& view, const VectorField& vf);
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
};

namespace Utilities {
template<>
struct Options::Option<SphereRenderer::Option::SPHERE_RADIUS> {
    float default_value = 0.2f;
};


template<>
struct Options::Option<SphereRenderer::Option::LEVEL_OF_DETAIL> {
    unsigned int default_value = 2;
};
}
}

#endif
