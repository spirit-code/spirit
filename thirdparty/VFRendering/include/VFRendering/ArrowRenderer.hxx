#ifndef VFRENDERING_ARROW_RENDERER_HXX
#define VFRENDERING_ARROW_RENDERER_HXX

#include <VFRendering/GlyphRenderer.hxx>

namespace VFRendering {
class ArrowRenderer : public GlyphRenderer {
public:
    enum Option {
        CONE_RADIUS = 200,
        CONE_HEIGHT,
        CYLINDER_RADIUS,
        CYLINDER_HEIGHT,
        LEVEL_OF_DETAIL
    };

    ArrowRenderer(const View& view, const VectorField& vf);
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
};

namespace Utilities {
template<>
struct Options::Option<ArrowRenderer::Option::CONE_RADIUS> {
    float default_value = 0.25f;
};


template<>
struct Options::Option<ArrowRenderer::Option::CONE_HEIGHT> {
    float default_value = 0.6f;
};


template<>
struct Options::Option<ArrowRenderer::Option::CYLINDER_RADIUS> {
    float default_value = 0.125f;
};


template<>
struct Options::Option<ArrowRenderer::Option::CYLINDER_HEIGHT> {
    float default_value = 0.7f;
};


template<>
struct Options::Option<ArrowRenderer::Option::LEVEL_OF_DETAIL> {
    unsigned int default_value = 20;
};
}
}

#endif
