#include "VFRendering/SphereRenderer.hxx"

#include <map>
#include <glm/glm.hpp>

namespace VFRendering {
static void setSphereMeshOptions(GlyphRenderer& renderer, const Options& options);

SphereRenderer::SphereRenderer(const View& view, const VectorField& vf) : GlyphRenderer(view, vf) {
  setSphereMeshOptions(*this, this->options());
}

void SphereRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case Option::SPHERE_RADIUS:
        case Option::LEVEL_OF_DETAIL:
            update_vertices = true;
            break;
        }
    }
    if (update_vertices) {
        setSphereMeshOptions(*this, options());
    }
    GlyphRenderer::optionsHaveChanged(changed_options);
}


static void setSphereMeshOptions(GlyphRenderer& renderer, const Options& options) {
    auto level_of_detail = options.get<SphereRenderer::Option::LEVEL_OF_DETAIL>();
    auto sphere_radius = options.get<SphereRenderer::Option::SPHERE_RADIUS>();

    // Enforce valid range
    if (sphere_radius < 0) {
        sphere_radius = 0;
    }
    std::vector<std::uint16_t> icosahedron_indices = {0, 1, 2, 1, 3, 2, 1, 4, 3, 5, 4, 1, 0, 5, 1, 0, 6, 5, 6, 7, 5, 5, 7, 4, 7, 8, 4, 7, 9, 8, 7, 6, 9, 6, 10, 9, 10, 11, 9, 10, 2, 11, 10, 0, 2, 6, 0, 10, 11, 2, 3, 8, 11, 3, 4, 8, 3, 11, 8, 9};
    std::vector<glm::vec3> icosahedron_positions = {
        {0.5257311121191336, 0, 0.8506508083520399},
        {0, 0.8506508083520399, 0.5257311121191336},
        {-0.5257311121191336, 0, 0.8506508083520399},
        {-0.8506508083520399, 0.5257311121191336, 0},
        {0, 0.8506508083520399, -0.5257311121191336},
        {0.8506508083520399, 0.5257311121191336, 0},
        {0.8506508083520399, -0.5257311121191336, 0},
        {0.5257311121191336, 0, -0.8506508083520399},
        {-0.5257311121191336, 0, -0.8506508083520399},
        {0, -0.8506508083520399, -0.5257311121191336},
        {0, -0.8506508083520399, 0.5257311121191336},
        {-0.8506508083520399, -0.5257311121191336, 0}
    };
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<std::uint16_t> indices;
    positions = icosahedron_positions;
    indices = icosahedron_indices;
    std::map<std::pair<std::uint16_t, std::uint16_t>, std::uint16_t> index_helper;
    std::vector<std::uint16_t> previous_indices = icosahedron_indices;
    for (unsigned int j = 0; j < level_of_detail; j++) {
        for (std::uint16_t i = 0; i < previous_indices.size()/3; i++) {
            std::uint16_t a_index  = previous_indices[3*i+0];
            std::uint16_t b_index  = previous_indices[3*i+1];
            std::uint16_t c_index  = previous_indices[3*i+2];
            glm::vec3 a = positions[a_index];
            glm::vec3 b = positions[b_index];
            glm::vec3 c = positions[c_index];
            glm::vec3 d = glm::normalize(a+b);
            glm::vec3 e = glm::normalize(b+c);
            glm::vec3 f = glm::normalize(c+a);
            std::uint16_t d_index = index_helper[{a_index, b_index}];
            std::uint16_t e_index = index_helper[{b_index, c_index}];
            std::uint16_t f_index = index_helper[{c_index, a_index}];
            if (d_index == 0) {
                d_index = std::uint16_t(positions.size());
                positions.push_back(d);
            }
            if (e_index == 0) {
                e_index = std::uint16_t(positions.size());
                positions.push_back(e);
            }
            if (f_index == 0) {
                f_index = std::uint16_t(positions.size());
                positions.push_back(f);
            }
            indices.push_back(a_index);
            indices.push_back(d_index);
            indices.push_back(f_index);
            indices.push_back(b_index);
            indices.push_back(e_index);
            indices.push_back(d_index);
            indices.push_back(c_index);
            indices.push_back(f_index);
            indices.push_back(e_index);
            indices.push_back(d_index);
            indices.push_back(e_index);
            indices.push_back(f_index);
        }
        previous_indices.swap(indices);
        indices.clear();
    }
    indices = previous_indices;
    normals = positions;
    for (auto& position : positions) {
        position = glm::normalize(position) * sphere_radius;
    }
    for (auto& normal : normals) {
        normal = glm::normalize(normal);
    }
    renderer.setGlyph(positions, normals, indices);
}
}
