#ifndef VFRENDERING_VECTORFIELD_HXX
#define VFRENDERING_VECTORFIELD_HXX

#include <array>
#include <memory>

#include <glm/glm.hpp>

#include <VFRendering/Options.hxx>
#include <VFRendering/FPSCounter.hxx>
#include <VFRendering/Utilities.hxx>
#include <VFRendering/Geometry.hxx>

namespace VFRendering {

class VectorField {
public:

    VectorField(const Geometry& geometry, const std::vector<glm::vec3>& vectors);
    virtual ~VectorField();

    void update(const Geometry& geometry, const std::vector<glm::vec3>& vectors);
    void updateGeometry(const Geometry& geometry);
    void updateVectors(const std::vector<glm::vec3>& vectors);

    const std::vector<glm::vec3>& positions() const;
    const std::vector<glm::vec3>& directions() const;
    const std::vector<std::array<Geometry::index_type, 3>>& surfaceIndices() const;
    const std::vector<std::array<Geometry::index_type, 4>>& volumeIndices() const;

    unsigned long geometryUpdateId() const;
    unsigned long vectorsUpdateId() const;
    
private:
    Geometry m_geometry;
    std::vector<glm::vec3> m_vectors;
    unsigned long m_geometry_update_id = 0;
    unsigned long m_vectors_update_id = 0;
};

}

#endif
