#ifndef VFRENDERING_VECTORFIELD_RENDERER_HXX
#define VFRENDERING_VECTORFIELD_RENDERER_HXX

#include <vector>

#include <glm/glm.hpp>

#include <VFRendering/View.hxx>
#include <VFRendering/VectorField.hxx>
#include <VFRendering/RendererBase.hxx>

namespace VFRendering {
class VectorFieldRenderer  : public RendererBase {
public:

    VectorFieldRenderer(const View& view, const VectorField& vf);

    virtual ~VectorFieldRenderer() {};
    virtual void updateIfNecessary();

protected:
    const std::vector<glm::vec3>& positions() const;
    const std::vector<glm::vec3>& directions() const;
    const std::vector<std::array<Geometry::index_type, 3>>& surfaceIndices() const;
    const std::vector<std::array<Geometry::index_type, 4>>& volumeIndices() const;

private:
    const VectorField& m_vf;
    unsigned long m_geometry_update_id = 0;
    unsigned long m_vectors_update_id = 0;
};

}

#endif
