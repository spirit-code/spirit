#include "VFRendering/VectorField.hxx"

#include <iostream>

namespace VFRendering {
VectorField::VectorField(const Geometry& geometry, const std::vector<glm::vec3>& vectors) : m_geometry(geometry), m_vectors(vectors) {
    m_vectors_update_id++;
    m_geometry_update_id++;
}

VectorField::~VectorField() {}

void VectorField::update(const Geometry& geometry, const std::vector<glm::vec3>& vectors) {
    m_geometry = geometry;
    m_vectors = vectors;
    m_vectors_update_id++;
    m_geometry_update_id++;
}

void VectorField::updateGeometry(const Geometry& geometry) {
    m_geometry = geometry;
    m_geometry_update_id++;
}

void VectorField::updateVectors(const std::vector<glm::vec3>& vectors) {
    m_vectors = vectors;
    m_vectors_update_id++;
}


unsigned long VectorField::geometryUpdateId() const {
    return m_geometry_update_id;
}

unsigned long VectorField::vectorsUpdateId() const {
    return m_vectors_update_id;
}

const std::vector<glm::vec3>& VectorField::positions() const {
    return m_geometry.positions();
}

const std::vector<glm::vec3>& VectorField::directions() const {
    return m_vectors;
}

const std::vector<std::array<Geometry::index_type, 3>>& VectorField::surfaceIndices() const {
    return m_geometry.surfaceIndices();
}

const std::vector<std::array<Geometry::index_type, 4>>& VectorField::volumeIndices() const {
    return m_geometry.volumeIndices();
}
}
