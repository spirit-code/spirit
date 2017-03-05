#ifndef VFRENDERING_ARROW_RENDERER_HXX
#define VFRENDERING_ARROW_RENDERER_HXX

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class ArrowRenderer : public RendererBase {
public:
    enum Option {
        CONE_RADIUS = 200,
        CONE_HEIGHT,
        CYLINDER_RADIUS,
        CYLINDER_HEIGHT,
        LEVEL_OF_DETAIL
    };

    ArrowRenderer(const View& view);
    virtual ~ArrowRenderer();
    virtual void update(bool keep_geometry) override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

private:
    void updateShaderProgram();
    void updateVertexData();
    void initialize();

    bool m_is_initialized = false;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    unsigned int m_ibo = 0;
    unsigned int m_instance_position_vbo = 0;
    unsigned int m_instance_direction_vbo = 0;
    unsigned int m_num_indices = 0;
    unsigned int m_num_instances = 0;
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
