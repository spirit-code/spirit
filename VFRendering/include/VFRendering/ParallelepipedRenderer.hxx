#ifndef VFRENDERING_PARALLELEPIPED_RENDERER_HXX
#define VFRENDERING_PARALLELEPIPED_RENDERER_HXX

#include <VFRendering/GlyphRenderer.hxx>

namespace VFRendering {

class ParallelepipedRenderer : public GlyphRenderer {
public:
    enum Option {
        LENGTH_A = 100,
        LENGTH_B,
        LENGTH_C
    };

    ParallelepipedRenderer(const View& view, const VectorField& vf);
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
};

namespace Utilities {

template<>
struct Options::Option<ParallelepipedRenderer::Option::LENGTH_A> {
    float default_value = 1.f;
};

template<>
struct Options::Option<ParallelepipedRenderer::Option::LENGTH_B> {
    float default_value = 1.f;
};

template<>
struct Options::Option<ParallelepipedRenderer::Option::LENGTH_C> {
    float default_value = 1.f;
};

}

}

#endif
