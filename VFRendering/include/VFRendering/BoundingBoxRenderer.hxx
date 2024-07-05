#ifndef VFRENDERING_BOUNDING_BOX_RENDERER_HXX
#define VFRENDERING_BOUNDING_BOX_RENDERER_HXX

#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class BoundingBoxRenderer : public RendererBase {
public:
    enum Option {
        COLOR = 600,
        LEVEL_OF_DETAIL,
        LINE_WIDTH
    };


    BoundingBoxRenderer(const View& view, const std::vector<glm::vec3>& vertices, const std::vector<float>& dashing_values={});
    static BoundingBoxRenderer forCuboid(const View& view, const glm::vec3& center, const glm::vec3& side_lengths, const glm::vec3& periodic_boundary_condition_lengths={0.0f, 0.0f, 0.0f}, float dashes_per_length=1.0f);
    static BoundingBoxRenderer forParallelepiped(const View& view, const glm::vec3& center, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const glm::vec3& periodic_boundary_condition_lengths={0.0f, 0.0f, 0.0f}, float dashes_per_length=1.0f);
    static BoundingBoxRenderer forHexagonalCell(const View& view, const glm::vec3& center, float radius, float height, const glm::vec2& periodic_boundary_condition_lengths={0.0f, 0.0f}, float dashes_per_length=1.0f);
    virtual ~BoundingBoxRenderer();
    virtual void draw(float aspect_ratio) override;
    virtual void optionsHaveChanged(const std::vector<int>& changed_options) override;

protected:
    virtual void update(bool keep_geometry) override;
    void updateVertexData();

private:
    void initialize();
    
    bool m_is_initialized = false;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    unsigned int m_dash_vbo = 0;
    unsigned int num_vertices = 0;
    std::vector<glm::vec3> m_vertices;
    std::vector<float> m_dashing_values;
};

namespace Utilities {
template<>
struct Options::Option<BoundingBoxRenderer::Option::COLOR>{
    glm::vec3 default_value = {1.0, 1.0, 1.0};
};

template<>
struct Options::Option<BoundingBoxRenderer::Option::LEVEL_OF_DETAIL>{
    int default_value = 10;
};

template<>
struct Options::Option<BoundingBoxRenderer::Option::LINE_WIDTH>{
    float default_value = 0.0;
};
}
}

#endif
