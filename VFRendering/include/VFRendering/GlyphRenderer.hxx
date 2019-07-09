#ifndef VFRENDERING_GLYPH_RENDERER_HXX
#define VFRENDERING_GLYPH_RENDERER_HXX

#include <VFRendering/VectorFieldRenderer.hxx>

namespace VFRendering {
class GlyphRenderer : public VectorFieldRenderer {
public:
    enum Option {
      ROTATE_GLYPHS = 1000
    };

    GlyphRenderer(const View& view, const VectorField& vf);
    virtual ~GlyphRenderer();
    virtual void update(bool keep_geometry) override;
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;
    void setGlyph(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals, const std::vector<std::uint16_t>& indices);

private:
    void updateShaderProgram();
    void updateVertexData();
    void initialize();

    bool m_is_initialized = false;
    std::vector<glm::vec3> m_positions;
    std::vector<glm::vec3> m_normals;
    std::vector<std::uint16_t> m_indices;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_position_vbo = 0;
    unsigned int m_normal_vbo = 0;
    unsigned int m_ibo = 0;
    unsigned int m_instance_position_vbo = 0;
    unsigned int m_instance_direction_vbo = 0;
    unsigned int m_num_indices = 0;
    unsigned int m_num_instances = 0;
};

namespace Utilities {
template<>
struct Options::Option<GlyphRenderer::Option::ROTATE_GLYPHS> {
    bool default_value = true;
};
}
}

#endif
