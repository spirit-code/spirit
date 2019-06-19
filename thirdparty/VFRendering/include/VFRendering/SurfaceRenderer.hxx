#ifndef VFRENDERING_SURFACE_RENDERER_HXX
#define VFRENDERING_SURFACE_RENDERER_HXX

#include <array>

#include <VFRendering/VectorFieldRenderer.hxx>

namespace VFRendering {

class SurfaceRenderer : public VectorFieldRenderer {
public:
    SurfaceRenderer(const View& view, const VectorField& vf);
    virtual ~SurfaceRenderer();
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

protected:
    virtual void update(bool keep_geometry) override;

private:
    void updateShaderProgram();
    void updateSurfaceIndices();
    void initialize();

    bool m_is_initialized = false;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_ibo = 0;
    unsigned int m_position_vbo = 0;
    unsigned int m_direction_vbo = 0;
    unsigned int m_num_indices = 0;
};

}

#endif
