#include "VFRendering/ArrowRenderer.hxx"

#include <glm/glm.hpp>

namespace VFRendering {
static void setArrowMeshOptions(GlyphRenderer& renderer, const Options& options);
ArrowRenderer::ArrowRenderer(const View& view, const VectorField& vf) : GlyphRenderer(view, vf) {
    setArrowMeshOptions(*this, this->options());
}

void ArrowRenderer::optionsHaveChanged(const std::vector<int>& changed_options) {
    bool update_vertices = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case Option::CONE_RADIUS:
        case Option::CONE_HEIGHT:
        case Option::CYLINDER_RADIUS:
        case Option::CYLINDER_HEIGHT:
        case Option::LEVEL_OF_DETAIL:
            update_vertices = true;
            break;
        }
    }
    if (update_vertices) {
        setArrowMeshOptions(*this, options());
    }
    GlyphRenderer::optionsHaveChanged(changed_options);
}


static void setArrowMeshOptions(GlyphRenderer& renderer, const Options& options) {
    auto level_of_detail = options.get<ArrowRenderer::Option::LEVEL_OF_DETAIL>();
    auto cone_height = options.get<ArrowRenderer::Option::CONE_HEIGHT>();
    auto cone_radius = options.get<ArrowRenderer::Option::CONE_RADIUS>();
    auto cylinder_height = options.get<ArrowRenderer::Option::CYLINDER_HEIGHT>();
    auto cylinder_radius = options.get<ArrowRenderer::Option::CYLINDER_RADIUS>();

    // Enforce valid range
    if (level_of_detail < 3) {
        level_of_detail = 3;
    }
    if (cone_height < 0) {
        cone_height = 0;
    }
    if (cone_radius < 0) {
        cone_radius = 0;
    }
    if (cylinder_height < 0) {
        cylinder_height = 0;
    }
    if (cylinder_radius < 0) {
        cylinder_radius = 0;
    }
    unsigned int i;
    double pi = 3.14159265358979323846;
    glm::vec3 baseNormal = {0, 0, -1};
    float z_offset = (cylinder_height - cone_height) / 2;
    float l = sqrt(cone_radius * cone_radius + cone_height * cone_height);
    float f1 = cone_radius / l;
    float f2 = cone_height / l;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    positions.reserve(level_of_detail * 5);
    normals.reserve(level_of_detail * 5);
    // The tip has no normal to prevent a discontinuity.
    positions.push_back({0, 0, z_offset + cone_height});
    normals.push_back({0, 0, 0});
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        positions.push_back({cone_radius* cos(alpha), cone_radius * sin(alpha), z_offset});
        normals.push_back({f2* cos(alpha), f2 * sin(alpha), f1});
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        positions.push_back({cone_radius* cos(alpha), cone_radius * sin(alpha), z_offset});
        normals.push_back(baseNormal);
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        positions.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset - cylinder_height});
        normals.push_back(baseNormal);
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        positions.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset - cylinder_height});
        normals.push_back({cos(alpha), sin(alpha), 0});
    }
    for (i = 0; i < level_of_detail; i++) {
        float alpha = 2 * pi * i / level_of_detail;
        positions.push_back({cylinder_radius* cos(alpha), cylinder_radius * sin(alpha), z_offset});
        normals.push_back({cos(alpha), sin(alpha), 0});
    }
    std::vector<std::uint16_t> indices;
    indices.reserve(level_of_detail * 15);
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(1 + i);
        indices.push_back(1 + (i + 1) % level_of_detail);
        indices.push_back(0);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail + 1);
        indices.push_back(level_of_detail + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail + 1 + i);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail * 2 + 1);
        indices.push_back(level_of_detail * 2 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 2 + 1 + i);
    }
    for (i = 0; i < level_of_detail; i++) {
        indices.push_back(level_of_detail * 3 + 1 + i);
        indices.push_back(level_of_detail * 3 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 4 + 1 + i);
        indices.push_back(level_of_detail * 4 + 1 + i);
        indices.push_back(level_of_detail * 3 + 1 + (i + 1) % level_of_detail);
        indices.push_back(level_of_detail * 4 + 1 + (i + 1) % level_of_detail);
    }
    renderer.setGlyph(positions, normals, indices);
}
}
