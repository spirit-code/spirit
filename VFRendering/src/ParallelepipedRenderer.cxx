#include "VFRendering/ParallelepipedRenderer.hxx"

#include <map>
#include <glm/glm.hpp>

namespace VFRendering {

static void setParallelepipedMeshOptions(GlyphRenderer& renderer, const Options& options);

ParallelepipedRenderer::ParallelepipedRenderer( const View& view, const VectorField& vf) 
    : GlyphRenderer(view, vf)
{
    setParallelepipedMeshOptions( *this, this->options() );
}

void ParallelepipedRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
            case Option::LENGTH_A:
            case Option::LENGTH_B:
            case Option::LENGTH_C:
                update_vertices = true;
                break;
        }
    }
    if (update_vertices) {
        setParallelepipedMeshOptions(*this, options());
    }
    GlyphRenderer::optionsHaveChanged(changed_options);
}

static void setParallelepipedMeshOptions(GlyphRenderer& renderer, const Options& options) {
    auto length_a = options.get<ParallelepipedRenderer::Option::LENGTH_A>();
    auto length_b = options.get<ParallelepipedRenderer::Option::LENGTH_B>();
    auto length_z = options.get<ParallelepipedRenderer::Option::LENGTH_C>();
    
    // Enforce valid range
    if (length_a < 0) length_a = 0;
    if (length_b < 0) length_b = 0;
    if (length_z < 0) length_z = 0;
    
    std::vector<std::uint16_t> cube_indices = 
        {0,3,1,1,3,2,
         4,5,7,5,7,6,
         8,10,9,8,11,10,
         12,14,13,12,15,14,
         16,19,17,17,19,18,
         20,21,22,22,23,20};

    std::vector<glm::vec3> cube_vertices = {
        {-1,-1,-1}, {-1,-1, 1}, { 1,-1, 1}, { 1,-1,-1},
        { 1,-1,-1}, { 1,-1, 1}, { 1, 1, 1}, { 1, 1,-1},
        { 1, 1,-1}, { 1, 1, 1}, {-1, 1, 1}, {-1, 1,-1},
        {-1, 1,-1}, {-1, 1, 1}, {-1,-1, 1}, {-1,-1,-1},
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
        { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}, {-1,-1, 1}
    };
    
    for (auto& vertex : cube_vertices) {
        vertex.x *= length_a;
        vertex.y *= length_b;
        vertex.z *= length_z;
    }

    std::vector<glm::vec3> normals = {
        {0,-1,0}, {0,-1,0}, {0,-1,0}, {0,-1,0}, 
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, 
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, 
        {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0}, 
        {0,0,-1}, {0,0,-1}, {0,0,-1}, {0,0,-1},
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}
    };

    for (auto& normal : normals) {
        normal = glm::normalize(normal);
    }

    renderer.setGlyph(cube_vertices, normals, cube_indices);
}

}
