#ifndef VFRENDERING_ISOSURFACE_RENDERER_HXX
#define VFRENDERING_ISOSURFACE_RENDERER_HXX

#include <functional>

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class IsosurfaceRenderer : public RendererBase {
public:

    typedef float isovalue_type;
    typedef std::function<isovalue_type(const glm::vec3&, const glm::vec3&)> value_function_type;

    enum Option {
        ISOVALUE = 700,
        VALUE_FUNCTION
    };

    IsosurfaceRenderer(const View& view);
    virtual ~IsosurfaceRenderer();
    virtual void update(bool keep_geometry) override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

private:
    void updateShaderProgram();
    void updateIsosurfaceIndices();
    void initialize();

    bool m_is_initialized = false;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_ibo = 0;
    unsigned int m_position_vbo = 0;
    unsigned int m_direction_vbo = 0;
    unsigned int m_num_indices = 0;

    bool m_value_function_changed;
    bool m_isovalue_changed;
};

namespace Utilities {
template<>
struct Options::Option<IsosurfaceRenderer::Option::ISOVALUE> {
    VFRendering::IsosurfaceRenderer::isovalue_type default_value = 0;
};

template<>
struct Options::Option<IsosurfaceRenderer::Option::VALUE_FUNCTION> {
    IsosurfaceRenderer::value_function_type default_value = [] (const glm::vec3& position, const glm::vec3& direction) {
        (void)position;
        return direction.z;
    };
};
}
}

#endif
