#ifndef VECTORFIELD_ISOSURFACE_HPP
#define VECTORFIELD_ISOSURFACE_HPP

#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

class VectorfieldIsosurface {
public:
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> directions;
  std::vector<glm::vec3> normals;
  std::vector<int> triangle_indices;
  static VectorfieldIsosurface calculate(const std::vector<glm::vec3>&, const std::vector<glm::vec3>& directions, const std::vector<float>& values, float isovalue, const std::vector<std::array<int, 4>>& tetrahedra);
};

#endif
