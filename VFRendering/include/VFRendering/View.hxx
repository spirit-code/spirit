#ifndef VFRENDERING_VIEW_HXX
#define VFRENDERING_VIEW_HXX

#include <array>
#include <memory>

#include <glm/glm.hpp>

#include <VFRendering/Options.hxx>
#include <VFRendering/FPSCounter.hxx>
#include <VFRendering/Utilities.hxx>
#include <VFRendering/Geometry.hxx>

namespace VFRendering {
class RendererBase;

enum class CameraMovementModes {
    TRANSLATE,
    ROTATE
};

enum class WidgetLocation {
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    TOP_LEFT,
    TOP_RIGHT
};

enum class VisualizationMode {
    SURFACE,
    ISOSURFACE,
    ARROWS,
    SPHERE
};


class View {
public:
    enum Option {
        BOUNDING_BOX_MIN,
        BOUNDING_BOX_MAX,
        SYSTEM_CENTER,
        VERTICAL_FIELD_OF_VIEW,
        BACKGROUND_COLOR,
        COLORMAP_IMPLEMENTATION,
        IS_VISIBLE_IMPLEMENTATION,
        CAMERA_POSITION,
        CENTER_POSITION,
        UP_VECTOR
    };

    View();
    virtual ~View();
    void draw();

    void mouseMove(const glm::vec2& position_before, const glm::vec2& position_after, CameraMovementModes mode);
    void mouseScroll(const float& wheel_delta);
    void setFramebufferSize(float width, float height);
    float getFramerate() const;

    void updateOptions(const Options& options);
    template<int index>
    void setOption(const typename Options::Type<index>::type& value);
    void options(const Options& options);
    const Options& options() const;

    void update(const Geometry& geometry, const std::vector<glm::vec3>& vectors);
    void updateVectors(const std::vector<glm::vec3>& vectors);

    const std::vector<glm::vec3>& positions() const;
    const std::vector<glm::vec3>& directions() const;
    const std::vector<std::array<Geometry::index_type, 3>>& surfaceIndices() const;
    const std::vector<std::array<Geometry::index_type, 4>>& volumeIndices() const;

    void renderers(const std::vector<std::pair<std::shared_ptr<RendererBase>, std::array<float, 4>>>& renderers);
    void renderers(const VisualizationMode& mode, bool show_bounding_box, bool show_miniview, WidgetLocation miniview_location, bool show_coordinate_system, WidgetLocation coordinate_system_location);

    unsigned long geometryUpdateId() const;
    unsigned long vectorsUpdateId() const;
    
private:
    void setCamera(glm::vec3 camera_position, glm::vec3 center_position, glm::vec3 up_vector);
    void optionsHaveChanged(const std::vector<int>& changed_options);
    void initialize();

    bool m_is_initialized = false;
    Geometry m_geometry;
    std::vector<glm::vec3> m_vectors;
    std::vector<std::pair<std::shared_ptr<RendererBase>, std::array<float, 4>>> m_renderers;
    Utilities::FPSCounter m_fps_counter;
    glm::vec2 m_framebuffer_size;
    bool m_is_centered = false;
    unsigned long m_geometry_update_id = 0;
    unsigned long m_vectors_update_id = 0;

    Options m_options;
};

template<int index>
void View::setOption(const typename Options::Type<index>::type& value) {
    updateOptions(Options::withOption<index>(value));
}

namespace Utilities {
/** Option to set the position of the lower left front corner of the bounding box. */
template<>
struct Options::Option<View::Option::BOUNDING_BOX_MIN> {
    glm::vec3 default_value = {-1, -1, -1};
};

/** Option to set the position of the upper right back corner of the bounding box. */
template<>
struct Options::Option<View::Option::BOUNDING_BOX_MAX> {
    glm::vec3 default_value = {1, 1, 1};
};

/** Option to set the position of the system center. */
template<>
struct Options::Option<View::Option::SYSTEM_CENTER> {
    glm::vec3 default_value = {0, 0, 0};
};

/** Option to set the vertical field of view for renderers using a perspective projection. */
template<>
struct Options::Option<View::Option::VERTICAL_FIELD_OF_VIEW> {
    float default_value = 45.0;
};

/** Option to set the background color. */
template<>
struct Options::Option<View::Option::BACKGROUND_COLOR> {
    glm::vec3 default_value = {0, 0, 0};
};

/** Option to set the GLSL code implementing the colormap function. */
template<>
struct Options::Option<View::Option::COLORMAP_IMPLEMENTATION> {
    std::string default_value = Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::DEFAULT);
};

/** Option to set the GLSL code implementing the is_visible function. */
template<>
struct Options::Option<View::Option::IS_VISIBLE_IMPLEMENTATION> {
    std::string default_value = "bool is_visible(vec3 position, vec3 direction) { return true; }";
};

/** Option to set the camera position. */
template<>
struct Options::Option<View::Option::CAMERA_POSITION> {
    glm::vec3 default_value = {14.5, 14.5, 30};
};

/** Option to set the camera focus center position. */
template<>
struct Options::Option<View::Option::CENTER_POSITION> {
    glm::vec3 default_value = {14.5, 14.5, 0};
};

/** Option to set the camera up vector. */
template<>
struct Options::Option<View::Option::UP_VECTOR> {
    glm::vec3 default_value = {0, 1, 0};
};
}
}

#endif
