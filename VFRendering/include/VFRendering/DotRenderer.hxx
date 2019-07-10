#ifndef VFRENDERING_DOT_RENDERER_HXX
#define VFRENDERING_DOT_RENDERER_HXX

#include <VFRendering/VectorFieldRenderer.hxx>

namespace VFRendering {
class DotRenderer : public VectorFieldRenderer {
public:
    
    enum Option {
        COLORMAP_IMPLEMENTATION = 140,
        IS_VISIBLE_IMPLEMENTATION,
        DOT_RADIUS,
        DOT_STYLE
    };

    enum DotStyle {
        CIRCLE,
        SQUARE
    };

    DotRenderer(const View& view, const VectorField& vf);
    virtual ~DotRenderer();
    virtual void update(bool keep_geometry) override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
    std::string getDotStyle(const DotStyle& dotstyle);

private:
    void updateShaderProgram();
    void updateVertexData();
    void initialize();

    bool m_is_initialized = false;
    
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    
    unsigned int m_instance_position_vbo = 0;
    unsigned int m_instance_direction_vbo = 0;
    
    unsigned int m_num_instances = 0;
};

namespace Utilities {

template<>
struct Options::Option<DotRenderer::Option::DOT_RADIUS> {
    float default_value = 1.f;
};

template<>
struct Options::Option<DotRenderer::Option::DOT_STYLE> {
    DotRenderer::DotStyle default_value = DotRenderer::DotStyle::CIRCLE;
};

// COLORMAP_IMPLEMENTATION & IS_VISIBLE_IMPLEMENTATION define in View.hxx

}

}
#endif
