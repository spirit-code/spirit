#ifndef VFRENDERING_VECTOR_SPHERE_RENDERER_HXX
#define VFRENDERING_VECTOR_SPHERE_RENDERER_HXX

#include <VFRendering/VectorFieldRenderer.hxx>

namespace VFRendering {

class VectorSphereRenderer : public VectorFieldRenderer {
public:
    enum Option {
        POINT_SIZE_RANGE = 400,
        INNER_SPHERE_RADIUS,
        USE_SPHERE_FAKE_PERSPECTIVE
    };

    VectorSphereRenderer(const View& view, const VectorField& vf);
    virtual ~VectorSphereRenderer();
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

protected:
    virtual void update(bool keep_geometry) override;

private:
    void updateShaderProgram();
    void initialize();
    
    bool m_is_initialized = false;
    unsigned int m_sphere_points_program = 0;
    unsigned int m_sphere_points_vao = 0;
    unsigned int m_sphere_points_positions_vbo = 0;
    unsigned int m_sphere_points_directions_vbo = 0;
    unsigned int m_sphere_background_program = 0;
    unsigned int m_sphere_background_vao = 0;
    unsigned int m_sphere_background_vbo = 0;
    unsigned int m_num_instances = 0;
};

namespace Utilities {
template<>
struct Options::Option<VectorSphereRenderer::Option::POINT_SIZE_RANGE> {
    glm::vec2 default_value = {1.0f, 4.0f};
};

template<>
struct Options::Option<VectorSphereRenderer::Option::INNER_SPHERE_RADIUS> {
    float default_value = 0.95f;
};

template<>
struct Options::Option<VectorSphereRenderer::Option::USE_SPHERE_FAKE_PERSPECTIVE> {
    bool default_value = true;
};
}
}

#endif
