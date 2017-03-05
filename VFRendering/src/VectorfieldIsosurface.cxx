#include "VectorfieldIsosurface.hxx"

#include <iostream>
#include <map>
#include <limits>

#include <glm/glm.hpp>

namespace VFRendering {
class VectorfieldIsosurfaceCalculation {
public:
    VectorfieldIsosurfaceCalculation(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& directions, const std::vector<float>& values, float isovalue) : in_positions(positions), in_directions(directions), in_values(values), in_isovalue(isovalue) {}

    void addTetrahedron(std::array<Geometry::index_type, 4> t);
    VectorfieldIsosurface getResultAndReset();

private:
    typedef Geometry::index_type index_type;
    typedef std::pair<index_type, index_type> edge_type;

    template<int NUM_INSIDE_POINTS>
    void generateTriangle(index_type i1, index_type i2, index_type i3, const std::array<glm::vec3, NUM_INSIDE_POINTS>& inside_points, bool flip_normal);

    index_type getIsopointIndex(const edge_type& edge);

    void generateOneTetrahedronTriangle(index_type in_i1, index_type out_i1, index_type out_i2, index_type out_i3, bool flip_normal);
    void generateTwoTetrahedronTriangles(index_type in_i1, index_type in_i2, index_type out_i1, index_type out_i2, bool flip_normal);

    // input
    const std::vector<glm::vec3>& in_positions;
    const std::vector<glm::vec3>& in_directions;
    const std::vector<float>& in_values;
    float in_isovalue;

    // internal
    std::map<edge_type, index_type> edge_indices;

    // output
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> directions;
    std::vector<glm::vec3> normals;
    std::vector<int> triangle_indices;
};


VectorfieldIsosurface VectorfieldIsosurfaceCalculation::getResultAndReset() {
    for (auto& normal : normals) {
        normal = glm::normalize(normal);
        if (glm::any(glm::isnan(normal))) {
            normal = glm::vec3(0, 0, 0);
        }
    }
    for (auto& direction : directions) {
        direction = glm::normalize(direction);
    }

    VectorfieldIsosurface isosurface;
    isosurface.positions.swap(positions);
    isosurface.directions.swap(directions);
    isosurface.normals.swap(normals);
    isosurface.triangle_indices.swap(triangle_indices);

    edge_indices.clear();
    positions.clear();
    directions.clear();
    normals.clear();
    triangle_indices.clear();
    return isosurface;
}

template<int NUM_INSIDE_POINTS>
void VectorfieldIsosurfaceCalculation::generateTriangle(index_type i1, index_type i2, index_type i3, const std::array<glm::vec3, NUM_INSIDE_POINTS>& inside_points, bool flip_normal) {
    glm::vec3 p1 = positions[i1];
    glm::vec3 p2 = positions[i2];
    glm::vec3 p3 = positions[i3];

    glm::vec3 n = glm::normalize(glm::cross(p2 - p1, p3 - p1));
    if (glm::any(glm::isnan(n))) {
        n = glm::vec3(0, 0, 0);
    }

    // In some cases one of the (up to two) inside points might be directly on
    // the triangle surface. In that case, if a second inside point is available
    // that other point should be used to determine whether the normal should
    // be flipped.
    float absmax_flip_normal_indicator = 0;
    for (const auto& inside_point : inside_points) {
        float flip_normal_indicator = glm::dot(n, inside_point - (p1 + p2 + p3) / 3.0f);
        if (glm::abs(flip_normal_indicator) > glm::abs(absmax_flip_normal_indicator)) {
            absmax_flip_normal_indicator = flip_normal_indicator;
        }
    }
    if (absmax_flip_normal_indicator > 0) {
        flip_normal = !flip_normal;
    }

    if (flip_normal) {
        n = -n;
        triangle_indices.push_back(i2);
        triangle_indices.push_back(i1);
        triangle_indices.push_back(i3);
    } else {
        triangle_indices.push_back(i1);
        triangle_indices.push_back(i2);
        triangle_indices.push_back(i3);
    }
    normals[i1] += n;
    normals[i2] += n;
    normals[i3] += n;
}

VectorfieldIsosurfaceCalculation::index_type VectorfieldIsosurfaceCalculation::getIsopointIndex(const VectorfieldIsosurfaceCalculation::edge_type& edge) {
    if (edge.first > edge.second) {
        return getIsopointIndex({edge.second, edge.first});
    }
    auto it = edge_indices.find(edge);
    if (it == edge_indices.end()) {
        float left_value = in_values[edge.first];
        float right_value = in_values[edge.second];
        float alpha;
        if (std::abs(left_value - right_value) < std::numeric_limits<float>::min()) {
            alpha = 0.5;
        } else {
            alpha = (in_isovalue - left_value) / (right_value - left_value);
            if (alpha < 0) {
                alpha = 0;
            } else if (alpha > 1) {
                alpha = 1;
            }
        }

        glm::vec3 left_point = in_positions[edge.first];
        glm::vec3 right_point = in_positions[edge.second];
        glm::vec3 isopoint = glm::mix(left_point, right_point, alpha);

        glm::vec3 left_direction = in_directions[edge.first];
        glm::vec3 right_direction = in_directions[edge.second];
        glm::vec3 isodirection = glm::normalize(glm::mix(left_direction, right_direction, alpha));

        index_type isopoint_index = positions.size();
        edge_indices[edge] = isopoint_index;
        positions.push_back(isopoint);
        directions.push_back(isodirection);
        normals.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
        return isopoint_index;
    } else {
        return it->second;
    }
}

void VectorfieldIsosurfaceCalculation::generateOneTetrahedronTriangle(index_type in_i1, index_type out_i1, index_type out_i2, index_type out_i3, bool flip_normal) {
    index_type i1 = getIsopointIndex({in_i1, out_i1});
    index_type i2 = getIsopointIndex({in_i1, out_i2});
    index_type i3 = getIsopointIndex({in_i1, out_i3});
    generateTriangle<1>(i1, i2, i3, {{in_positions[in_i1]}}, flip_normal);
}

void VectorfieldIsosurfaceCalculation::generateTwoTetrahedronTriangles(index_type in_i1, index_type in_i2, index_type out_i1, index_type out_i2, bool flip_normal) {
    index_type i1 = getIsopointIndex({in_i1, out_i1});
    index_type i2 = getIsopointIndex({in_i1, out_i2});
    index_type i3 = getIsopointIndex({in_i2, out_i1});
    index_type i4 = getIsopointIndex({in_i2, out_i2});
    generateTriangle<2>(i1, i4, i2, {{in_positions[in_i1], in_positions[in_i2]}}, flip_normal);
    generateTriangle<2>(i1, i4, i3, {{in_positions[in_i1], in_positions[in_i2]}}, flip_normal);
}

void VectorfieldIsosurfaceCalculation::addTetrahedron(std::array<Geometry::index_type, 4> t) {
    int index = 0;
    for (int i = 0; i < 4; i++) {
        if (in_values[t[i]] > in_isovalue) {
            index += (1 << i);
        }
    }
    bool flip_normal = (index >= 8);
    if (flip_normal) {
        index = 15 - index;
    }
    index_type in_i1 = -1;
    index_type in_i2 = -1;
    index_type out_i1 = -1;
    index_type out_i2 = -1;
    index_type out_i3 = -1;
    index_type result_tri = 1;
    switch (index) {
    case 0:
        return;
    case 1:
        in_i1 = t[0];
        out_i1 = t[1];
        out_i2 = t[2];
        out_i3 = t[3];
        break;
    case 2:
        in_i1 = t[1];
        out_i1 = t[0];
        out_i2 = t[2];
        out_i3 = t[3];
        break;
    case 3:
        in_i1 = t[0];
        in_i2 = t[1];
        out_i1 = t[2];
        out_i2 = t[3];
        result_tri = 2;
        break;
    case 4:
        in_i1 = t[2];
        out_i1 = t[0];
        out_i2 = t[1];
        out_i3 = t[3];
        break;
    case 5:
        in_i1 = t[0];
        in_i2 = t[2];
        out_i1 = t[1];
        out_i2 = t[3];
        result_tri = 2;
        break;
    case 6:
        in_i1 = t[1];
        in_i2 = t[2];
        out_i1 = t[0];
        out_i2 = t[3];
        result_tri = 2;
        break;
    case 7:
        flip_normal = !flip_normal;
        in_i1 = t[3];
        out_i1 = t[0];
        out_i2 = t[1];
        out_i3 = t[2];
        break;
    }
    if (result_tri == 1) {
        if (in_values[in_i1] == in_isovalue) {
            return;
        }
        generateOneTetrahedronTriangle(in_i1, out_i1, out_i2, out_i3, flip_normal);
    } else {
        generateTwoTetrahedronTriangles(in_i1, in_i2, out_i1, out_i2, flip_normal);
    }
}

VectorfieldIsosurface VectorfieldIsosurface::calculate(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& directions, const std::vector<float>& values, float isovalue, const std::vector<std::array<Geometry::index_type, 4>>& tetrahedra) {
    VectorfieldIsosurfaceCalculation calculation(positions, directions, values, isovalue);
    for (auto& t : tetrahedra) {
        calculation.addTetrahedron(t);
    }
    return calculation.getResultAndReset();
}
}
