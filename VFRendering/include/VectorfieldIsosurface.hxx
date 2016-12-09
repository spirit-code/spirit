#ifndef VFRENDERING_VECTORFIELD_ISOSURFACE_HXX
#define VFRENDERING_VECTORFIELD_ISOSURFACE_HXX

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include <VFRendering/Geometry.hxx>

namespace VFRendering {
class VectorfieldIsosurface {
public:
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> directions;
    std::vector<glm::vec3> normals;
    std::vector<int> triangle_indices;
    static VectorfieldIsosurface calculate(const std::vector<glm::vec3>&, const std::vector<glm::vec3>& directions, const std::vector<float>& values, float isovalue, const std::vector<std::array<Geometry::index_type, 4>>& tetrahedra);
};
}

#endif
