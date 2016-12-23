#ifndef VFRENDERING_COORDINATE_SYSTEM_RENDERER_HXX
#define VFRENDERING_COORDINATE_SYSTEM_RENDERER_HXX

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class CoordinateSystemRenderer : public RendererBase {
public:
    enum Option {
        AXIS_LENGTH = 500,
        ORIGIN,
        LEVEL_OF_DETAIL,
        CONE_HEIGHT,
        CONE_RADIUS,
        CYLINDER_HEIGHT,
        CYLINDER_RADIUS,
        NORMALIZE
    };

    CoordinateSystemRenderer(const View& view);
    virtual ~CoordinateSystemRenderer();
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

protected:
    virtual void update(bool keep_geometry) override;

private:
    void updateShaderProgram();
    void updateVertexData();
    void initialize();

    bool m_is_initialized = false;

    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    unsigned int m_num_vertices = 0;
};

namespace Utilities {
template<>
struct Options::Option<CoordinateSystemRenderer::Option::AXIS_LENGTH> {
    glm::vec3 default_value = {0.5, 0.5, 0.5};
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::ORIGIN> {
    glm::vec3 default_value = {0.0, 0.0, 0.0};
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::CONE_RADIUS> {
    float default_value = 0.1f;
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::CONE_HEIGHT> {
    float default_value = 0.3f;
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::CYLINDER_RADIUS> {
    float default_value = 0.07f;
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::CYLINDER_HEIGHT> {
    float default_value = 0.7f;
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::LEVEL_OF_DETAIL> {
    unsigned int default_value = 100;
};


template<>
struct Options::Option<CoordinateSystemRenderer::Option::NORMALIZE> {
    bool default_value = false;
};
}
}

#endif
