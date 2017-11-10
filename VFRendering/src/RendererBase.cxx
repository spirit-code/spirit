#include "VFRendering/RendererBase.hxx"

namespace VFRendering {
RendererBase::RendererBase(const View& view) : m_view(view), m_options(m_view.options()) {}

const Options& RendererBase::options() const {
    return m_options;
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

void RendererBase::updateIfNecessary() { }

}
