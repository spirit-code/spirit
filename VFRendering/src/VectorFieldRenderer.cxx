#include "VFRendering/VectorFieldRenderer.hxx"

namespace VFRendering {
VectorFieldRenderer::VectorFieldRenderer(const View& view, const VectorField& vf)  : RendererBase(view), m_vf(vf) {}

const std::vector<glm::vec3>& VectorFieldRenderer::positions() const {
    return m_vf.positions();
}

const std::vector<glm::vec3>& VectorFieldRenderer::directions() const {
    return m_vf.directions();
}

const std::vector<std::array<Geometry::index_type, 3>>& VectorFieldRenderer::surfaceIndices() const {
    return m_vf.surfaceIndices();
}

const std::vector<std::array<Geometry::index_type, 4>>& VectorFieldRenderer::volumeIndices() const {
    return m_vf.volumeIndices();
}

void VectorFieldRenderer::updateIfNecessary() {
    if (m_geometry_update_id != m_vf.geometryUpdateId()) {
        update(false);
        m_geometry_update_id = m_vf.geometryUpdateId();
        m_vectors_update_id = m_vf.vectorsUpdateId();
    } else if (m_vectors_update_id != m_vf.vectorsUpdateId()) {
        update(true);
        m_vectors_update_id = m_vf.vectorsUpdateId();
    }
}

}
