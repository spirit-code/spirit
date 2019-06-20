#include "VFRendering/Geometry.hxx"

#ifndef NO_QHULL
#include "Qhull.h"
#include "QhullFacetList.h"
#include "QhullVertexSet.h"
#endif

namespace VFRendering {
Geometry::Geometry() {}

Geometry::Geometry(const std::vector<glm::vec3>& positions, const std::vector<std::array<Geometry::index_type, 3>>& surface_indices, const std::vector<std::array<Geometry::index_type, 4>>& volume_indices, const bool& is_2d) : m_positions(positions), m_surface_indices(surface_indices), m_volume_indices(volume_indices), m_is_2d(is_2d) {}

const std::vector<glm::vec3>& Geometry::positions() const {
    return m_positions;
}

const std::vector<std::array<Geometry::index_type, 3>>& Geometry::surfaceIndices() const {
    if (m_surface_indices.empty() && m_positions.size() >= 3) {
        // TODO: calculate surface triangulation
        std::cerr << "surface indices calculation for arbitrary geometry is not implemented yet" << std::endl;
    }
    return m_surface_indices;
}

const bool& Geometry::is2d() const {
    return m_is_2d;
}

const glm::vec3& Geometry::min() const {
    if (!m_bounds_min_set) {
        m_bounds_min_set = true;
        if (positions().size() > 0) {
            glm::vec3 m_bounds_min = positions()[0];
            for (const auto& position : positions()) {
                m_bounds_min = glm::min(m_bounds_min, position);
            }
        } else {
            // origin as fallback
            m_bounds_min = glm::vec3(0, 0, 0);
        }
    }
    return m_bounds_min;
}

const glm::vec3& Geometry::max() const {
    if (!m_bounds_max_set) {
        m_bounds_max_set = true;
        if (positions().size() > 0) {
            glm::vec3 m_bounds_max = positions()[0];
            for (const auto& position : positions()) {
                m_bounds_max = glm::max(m_bounds_max, position);
            }
        } else {
            // origin as fallback
            m_bounds_max = glm::vec3(0, 0, 0);
        }
    }
    return m_bounds_max;
}

const std::vector<std::array<Geometry::index_type, 4>>& Geometry::volumeIndices() const {
    if (m_volume_indices.empty() && m_positions.size() >= 4 && !is2d()) {
        // calculate the volume indices using QHull

#ifdef NO_QHULL
        std::cerr << "volume indices calculation for arbitrary geometry required QHull" << std::endl;
#else
        // QHull requires double precision floating point numbers
        const std::vector<glm::dvec3> dpositions(m_positions.cbegin(), m_positions.cend());

        ::orgQhull::Qhull qhull;
        qhull.runQhull("", 3, dpositions.size(), &(dpositions[0].x), "qhull d Qt Qbb Qz");

        // copy results
        for (auto facet : qhull.facetList()) {
            if (!facet.isUpperDelaunay()) {
                std::array<Geometry::index_type, 4> tetrahedron;
                auto vertices = facet.vertices();
                for (int i = 0; i < 4; i++) {
                    tetrahedron[i] = vertices[i].point().id();
                }
                m_volume_indices.push_back(tetrahedron);
            }
        }
#endif
    }
    return m_volume_indices;
}

Geometry Geometry::cartesianGeometry(glm::ivec3 n, glm::vec3 bounds_min, glm::vec3 bounds_max) {
    std::vector<float> xs(n.x);
    std::vector<float> ys(n.y);
    std::vector<float> zs(n.z);
    for (int i = 0; i < n.x; i++) {
        xs[i] = i / (n.x - 1.0) * (bounds_max.x - bounds_min.x) + bounds_min.x;
    }
    for (int i = 0; i < n.x; i++) {
        ys[i] = i / (n.y - 1.0) * (bounds_max.y - bounds_min.y) + bounds_min.y;
    }
    for (int i = 0; i < n.x; i++) {
        zs[i] = i / (n.z - 1.0) * (bounds_max.z - bounds_min.z) + bounds_min.z;
    }
    return Geometry::rectilinearGeometry(xs, ys, zs);
}

Geometry Geometry::rectilinearGeometry(const std::vector<float>& xs, const std::vector<float>& ys, const std::vector<float>& zs) {
    glm::ivec3 n(xs.size(), ys.size(), zs.size());

    if (n.x < 1 || n.y < 1 || n.z < 1) {
        return Geometry({}, {}, {}, false);
    }

    bool is_2d = (n.x == 1 || n.y == 1 || n.z == 1);

    // TODO: allow different offsets
    Geometry::index_type x_offset = 1;
    Geometry::index_type y_offset = n.x;
    Geometry::index_type z_offset = n.x * n.y;

    std::vector<glm::vec3> positions;
    positions.reserve(n.x * n.y * n.z);
    for (int iz = 0; iz < n.z; iz++) {
        for (int iy = 0; iy < n.y; iy++) {
            for (int ix = 0; ix < n.x; ix++) {
                positions.push_back({xs[ix], ys[iy], zs[iz]});
            }
        }
    }

    std::vector<std::array<Geometry::index_type, 3>> surface_indices;
    if (is_2d) {
        Geometry::index_type offsets[] = {
            0, x_offset, y_offset, y_offset, x_offset, x_offset + y_offset
        };
        for (int ix = 0; ix < n.x - 1; ix++) {
            for (int iy = 0; iy < n.y - 1; iy++) {
                Geometry::index_type base_index = ix * x_offset + iy * y_offset;
                std::array<Geometry::index_type, 3> left_triangle = {
                    {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                };
                surface_indices.push_back(left_triangle);
                std::array<Geometry::index_type, 3> right_triangle = {
                    {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                };
                surface_indices.push_back(right_triangle);
            }
        }
    } else {
        // bottom and top
        {
            Geometry::index_type offsets[] = {
                0, x_offset, y_offset, y_offset, x_offset, x_offset + y_offset
            };
            for (int ix = 0; ix < n.x - 1; ix++) {
                for (int iy = 0; iy < n.y - 1; iy++) {
                    Geometry::index_type base_index = ix * x_offset + iy * y_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
            for (int ix = 0; ix < n.x - 1; ix++) {
                for (int iy = 0; iy < n.y - 1; iy++) {
                    Geometry::index_type base_index = ix * x_offset + iy * y_offset + (n.z - 1) * z_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
        }
        // front and back
        {
            Geometry::index_type offsets[] = {
                0, x_offset, z_offset, z_offset, x_offset, x_offset + z_offset
            };
            for (int ix = 0; ix < n.x - 1; ix++) {
                for (int iz = 0; iz < n.z - 1; iz++) {
                    Geometry::index_type base_index = ix * x_offset + iz * z_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
            for (int ix = 0; ix < n.x - 1; ix++) {
                for (int iz = 0; iz < n.z - 1; iz++) {
                    Geometry::index_type base_index = ix * x_offset + (n.y - 1) * y_offset + iz * z_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
        }
        // left and right
        {
            Geometry::index_type offsets[] = {
                0, y_offset, z_offset, z_offset, y_offset, y_offset + z_offset
            };
            for (int iy = 0; iy < n.y - 1; iy++) {
                for (int iz = 0; iz < n.z - 1; iz++) {
                    Geometry::index_type base_index = iy * y_offset + iz * z_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
            for (int iy = 0; iy < n.y - 1; iy++) {
                for (int iz = 0; iz < n.z - 1; iz++) {
                    Geometry::index_type base_index = (n.x - 1) * x_offset + iy * y_offset + iz * z_offset;
                    std::array<Geometry::index_type, 3> left_triangle = {
                        {base_index + offsets[0], base_index + offsets[1], base_index + offsets[2]}
                    };
                    surface_indices.push_back(left_triangle);
                    std::array<Geometry::index_type, 3> right_triangle = {
                        {base_index + offsets[3], base_index + offsets[4], base_index + offsets[5]}
                    };
                    surface_indices.push_back(right_triangle);
                }
            }
        }
    }

    std::vector<std::array<Geometry::index_type, 4>> volume_indices;
    if (is_2d) {
        // 2d geometry has no volume, so volume_indices should be empty
    } else {
        int cell_indices[] = {
            0, 1, 5, 3,
            1, 3, 2, 5,
            3, 2, 5, 6,
            7, 6, 5, 3,
            4, 7, 5, 3,
            0, 4, 3, 5
        };

        Geometry::index_type offsets[] = {
            0, x_offset, x_offset + y_offset, y_offset,
            z_offset, x_offset + z_offset, x_offset + y_offset + z_offset, y_offset + z_offset
        };

        for (int ix = 0; ix < n.x - 1; ix++) {
            for (int iy = 0; iy < n.y - 1; iy++) {
                for (int iz = 0; iz < n.z - 1; iz++) {
                    Geometry::index_type base_index = ix * x_offset + iy * y_offset + iz * z_offset;
                    for (int j = 0; j < 6; j++) {
                        std::array<Geometry::index_type, 4> tetrahedron;
                        for (int k = 0; k < 4; k++) {
                            Geometry::index_type index = base_index + offsets[cell_indices[j * 4 + k]];
                            tetrahedron[k] = index;
                        }
                        volume_indices.push_back(tetrahedron);
                    }
                }
            }
        }
    }
    Geometry result = Geometry(positions, surface_indices, volume_indices, is_2d);
    result.m_bounds_min = {xs[0], ys[0], zs[0]};
    result.m_bounds_min_set = true;
    result.m_bounds_max = {xs[n.x - 1], ys[n.y - 1], zs[n.z - 1]};
    result.m_bounds_max_set = true;
    return result;
}
}
