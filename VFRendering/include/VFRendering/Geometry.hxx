#ifndef VFRENDERING_GEOMETRY_HXX
#define VFRENDERING_GEOMETRY_HXX

#include <vector>
#include <array>

#include <glm/glm.hpp>

namespace VFRendering {
class Geometry {
public:
    /** Type for use as indices into positions and vectors.
     *
     *  Intuitively, this should be std::vector<glm::vec3>::size_type. However,
     *  as OpenGL does not support unsigned long int values for indices, it is
     *  defined as unsigned int instead.
     *
     *  This imposes a limit on the maximum number of positions/vectors. For
     *  this to be a problem, the directions and vectors would take up at least
     *  96 GiB of GPU memory, so it should be fine for the next years.
     */
    typedef unsigned int index_type;

    Geometry();
    Geometry(const std::vector<glm::vec3>& positions, const std::vector<std::array<index_type, 3>>& surface_indices={}, const std::vector<std::array<index_type, 4>>& volume_indices={}, const bool& is_2d=false);

    const std::vector<glm::vec3>& positions() const;
    const std::vector<std::array<index_type, 3>>& surfaceIndices() const;
    const std::vector<std::array<index_type, 4>>& volumeIndices() const;
    const glm::vec3& min() const;
    const glm::vec3& max() const;
    const bool& is2d() const;

    static Geometry cartesianGeometry(glm::ivec3 n, glm::vec3 bounds_min, glm::vec3 bounds_max);
    static Geometry rectilinearGeometry(const std::vector<float>& xs, const std::vector<float>& ys, const std::vector<float>& zs);

private:
    std::vector<glm::vec3> m_positions;
    mutable std::vector<std::array<index_type, 3>> m_surface_indices;
    mutable std::vector<std::array<index_type, 4>> m_volume_indices;
    bool m_is_2d;
    mutable bool m_bounds_min_set;
    mutable glm::vec3 m_bounds_min;
    mutable bool m_bounds_max_set;
    mutable glm::vec3 m_bounds_max;
};
}

#endif
