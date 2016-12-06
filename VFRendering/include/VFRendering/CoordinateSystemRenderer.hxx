#ifndef VFRENDERING_COORDINATE_SYSTEM_RENDERER_HXX
#define VFRENDERING_COORDINATE_SYSTEM_RENDERER_HXX

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class CoordinateSystemRenderer : public RendererBase {
public:
    enum Option {
        AXIS_LENGTH = 500, ORIGIN
    };

    CoordinateSystemRenderer(const View& view);
    virtual ~CoordinateSystemRenderer();
    virtual void update(bool keep_geometry) override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

private:
    void updateShaderProgram();
    void initialize();

    bool m_is_initialized = false;

    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
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
}
}

#endif
