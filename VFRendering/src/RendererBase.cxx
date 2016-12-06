#include "VFRendering/RendererBase.hxx"

namespace VFRendering {
RendererBase::RendererBase(const View& view) : m_view(view), m_options(view.options()) {}

const Options& RendererBase::options() const {
    return m_options;
}

const std::vector<glm::vec3>& RendererBase::positions() const {
    return m_view.positions();
}

const std::vector<glm::vec3>& RendererBase::directions() const {
    return m_view.directions();
}

const std::vector<std::array<Geometry::index_type, 3>>& RendererBase::surfaceIndices() const {
    return m_view.surfaceIndices();
}

const std::vector<std::array<Geometry::index_type, 4>>& RendererBase::volumeIndices() const {
    return m_view.volumeIndices();
}

void RendererBase::options(const Options& options) {
    m_options = Options();
    updateOptions(options);
}

void RendererBase::optionsHaveChanged(const std::vector<int>& changed_options) {
    (void)changed_options;
}

void RendererBase::updateOptions(const Options& options) {
    auto changed_options = m_options.update(options);
    if (changed_options.size() == 0) {
        return;
    }
    optionsHaveChanged(changed_options);
}

void RendererBase::updateIfNecessary() {
    if (m_geometry_update_id != m_view.geometryUpdateId()) {
        update(false);
        m_geometry_update_id = m_view.geometryUpdateId();
        m_vectors_update_id = m_view.vectorsUpdateId();
    } else if (m_vectors_update_id != m_view.vectorsUpdateId()) {
        update(true);
        m_vectors_update_id = m_view.vectorsUpdateId();
    }
}

}
