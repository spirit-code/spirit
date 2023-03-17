#pragma once
#ifndef SPIRIT_SPINWIDGET_HPP
#define SPIRIT_SPINWIDGET_HPP

#include "MouseDecoratorWidget.hpp"

#include <QtWidgets/QOpenGLWidget>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/RendererBase.hxx>
#include <VFRendering/SurfaceRenderer.hxx>
#include <VFRendering/VectorSphereRenderer.hxx>
#include <VFRendering/View.hxx>

#include "glm/glm.hpp"

#include <memory>
#include <set>

struct State;

class SpinWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    enum class Colormap
    {
        HSV,
        HSV_NO_Z,
        BLUE_WHITE_RED,
        BLUE_GREEN_RED,
        BLUE_RED,
        WHITE,
        GRAY,
        BLACK,
        OTHER
    };

    enum class Color
    {
        BLACK,
        GRAY,
        WHITE,
        OTHER
    };

    enum class WidgetLocation
    {
        BOTTOM_LEFT,
        BOTTOM_RIGHT,
        TOP_LEFT,
        TOP_RIGHT
    };

    enum class VisualizationMode
    {
        SYSTEM,
        SPHERE
    };

    enum class VisualizationSource
    {
        SPINS,
        EFF_FIELD
    };

    enum class InteractionMode
    {
        REGULAR,
        DRAG,
        DEFECT,
        PIN
    };

    enum class SystemMode
    {
        CUSTOM,
        ISOSURFACE,
        SLAB_X,
        SLAB_Y,
        SLAB_Z
    };

    SpinWidget( std::shared_ptr<State> state, QWidget * parent = 0 );
    void setSuspended( bool suspended );
    void updateData( bool update_directions = true, bool update_geometry = true, bool update_camera = true );
    void updateVectorFieldDirections();
    void updateVectorFieldGeometry();
    void initializeGL();
    void resizeGL( int width, int height );
    void paintGL();
    void screenShot( std::string filename );
    float getFramesPerSecond() const;

    void setVisualisationSource( int source );
    int m_source;

    const VFRendering::View * view();
    const VFRendering::VectorField * vectorfield();

    void addIsosurface( std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer );
    void removeIsosurface( std::shared_ptr<VFRendering::IsosurfaceRenderer> );

    // --- Mode
    int visualisationNCellSteps();
    void setVisualisationNCellSteps( int n_cell_steps );
    void setVisualizationMode( SpinWidget::VisualizationMode visualization_mode );
    SpinWidget::VisualizationMode visualizationMode();
    SpinWidget::VisualizationMode visMode;
    void setInteractionMode( SpinWidget::InteractionMode mode );
    SpinWidget::InteractionMode interactionMode();
    bool show_miniview, show_coordinatesystem;
    // --- MiniView
    void setVisualizationMiniview( bool show, SpinWidget::WidgetLocation location );
    bool isMiniviewEnabled() const;
    void enableMiniview( bool enabled );
    WidgetLocation miniviewPosition() const;
    void setMiniviewPosition( WidgetLocation position );
    // --- Coordinate System
    void setVisualizationCoordinatesystem( bool show, SpinWidget::WidgetLocation location );
    bool isCoordinateSystemEnabled() const;
    void enableCoordinateSystem( bool enabled );
    WidgetLocation coordinateSystemPosition() const;
    void setCoordinateSystemPosition( WidgetLocation position );

    // --- System
    void enableSystem( bool arrows, bool boundingbox, bool surface, bool isosurface );
    void cycleSystem( bool forward = true );
    void cycleSystem( SystemMode mode );
    SystemMode systemCycle();
    void moveSlab( int amount );
    bool show_arrows, show_boundingbox, show_surface, show_isosurface;
    //    Arrows
    void setArrows( float size = 1, int lod = 20 );
    float arrowSize() const;
    int arrowLOD() const;
    glm::vec2 xRangeDirection() const;
    glm::vec2 yRangeDirection() const;
    glm::vec2 zRangeDirection() const;
    void setOverallDirectionRange( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range );
    glm::vec2 xRangePosition() const;
    glm::vec2 yRangePosition() const;
    glm::vec2 zRangePosition() const;

    void
    setCellFilter( int cell_a_min, int cell_a_max, int cell_b_min, int cell_b_max, int cell_c_min, int cell_c_max );

    std::array<int, 6> getCellFilter()
    {
        return std::array<int, 6>{ m_cell_a_min, m_cell_a_max, m_cell_b_min, m_cell_b_max, m_cell_c_min, m_cell_c_max };
    }

    void setOverallPositionRange( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range );
    void updateIsVisibleImplementation();
    //    Bounding Box
    bool isBoundingBoxEnabled() const;
    void enableBoundingBox( bool enabled );
    //    Surface
    void setSurface( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range );
    glm::vec2 surfaceXRange() const;
    glm::vec2 surfaceYRange() const;
    glm::vec2 surfaceZRange() const;

    // --- Sphere
    glm::vec2 spherePointSizeRange() const;
    void setSpherePointSizeRange( glm::vec2 sphere_point_size_range );

    // --- Colors
    Colormap colormap_general() const;
    Colormap colormap_arrows() const;
    void setColormapGeneral( Colormap colormap );
    void setColormapArrows( Colormap colormap );
    float colormap_rotation();
    std::array<bool, 2> colormap_inverted();
    glm::vec3 colormap_cardinal_a();
    glm::vec3 colormap_cardinal_b();
    glm::vec3 colormap_cardinal_c();
    void setColormapRotationInverted(
        int phi, bool invert_z, bool invert_xy, glm::vec3 cardinal_a = { 1, 0, 0 }, glm::vec3 cardinal_b = { 0, 1, 0 },
        glm::vec3 cardinal_c = { 0, 0, 1 } );
    std::string getColormapRotationInverted(
        Colormap colormap, int phi = 0, bool invert_z = false, bool invert_xy = false,
        glm::vec3 cardinal_a = { 1, 0, 0 }, glm::vec3 cardinal_b = { 0, 1, 0 }, glm::vec3 cardinal_c = { 0, 0, 1 } );
    Color backgroundColor() const;
    void setBackgroundColor( Color background_color );
    Color boundingBoxColor() const;
    void setBoundingBoxColor( Color bounding_box_color );
    void updateBoundingBoxIndicators();

    // --- Camera
    void cycleCamera();
    void setCameraToDefault();
    void setCameraToX( bool inverted = false );
    void setCameraToY( bool inverted = false );
    void setCameraToZ( bool inverted = false );
    void setCameraPosition( const glm::vec3 & camera_position );
    void setCameraFocus( const glm::vec3 & center_position );
    void setCameraUpVector( const glm::vec3 & up_vector );
    glm::vec3 getCameraPositon();
    glm::vec3 getCameraFocus();
    glm::vec3 getCameraUpVector();
    float verticalFieldOfView() const;
    void setVerticalFieldOfView( float vertical_field_of_view );
    bool cameraProjection();
    void setCameraProjection( bool perspective );
    // --- Move Camera
    void moveCamera( float backforth, float rightleft, float updown );
    void rotateCamera( float theta, float phi );
    bool getCameraRotationType();
    void setCameraRotationType( bool free );

    // --- Light
    void setLightPosition( float theta, float phi );
    std::array<float, 2> getLightPosition();

    void setPasteAtomType( int type );

protected:
    virtual void mouseMoveEvent( QMouseEvent * event );
    virtual void mousePressEvent( QMouseEvent * event );
    virtual void mouseReleaseEvent( QMouseEvent * event );
    virtual void wheelEvent( QWheelEvent * event );
    void closeEvent( QCloseEvent * event );

protected slots:
    void teardownGL();

private:
    std::shared_ptr<State> state;
    QPoint m_previous_mouse_position;
    bool _reset_camera;
    bool m_camera_rotate_free;
    bool m_camera_projection_perspective;
    float m_light_theta, m_light_phi;
    int paste_atom_type;

    // temporaries for system cycle
    void setSystemCycle( SystemMode mode );
    void setSlabRanges();
    int idx_cycle;
    bool user_show_arrows, user_show_boundingbox, user_show_surface, user_show_isosurface;
    float user_fov;
    glm::vec3 slab_displacements;

    // Renderers
    std::shared_ptr<VFRendering::RendererBase> m_mainview;
    std::shared_ptr<VFRendering::RendererBase> m_miniview;
    WidgetLocation m_location_miniview;
    std::shared_ptr<VFRendering::CoordinateSystemRenderer> m_coordinatesystem;
    WidgetLocation m_location_coordinatesystem;
    std::shared_ptr<VFRendering::VectorSphereRenderer> m_sphere;

    std::shared_ptr<VFRendering::CombinedRenderer> m_system;
    std::shared_ptr<VFRendering::ArrowRenderer> m_renderer_arrows;
    std::shared_ptr<VFRendering::BoundingBoxRenderer> m_renderer_boundingbox;
    std::shared_ptr<VFRendering::RendererBase> m_renderer_surface;
    std::shared_ptr<VFRendering::IsosurfaceRenderer> m_renderer_surface_3D;
    std::shared_ptr<VFRendering::SurfaceRenderer> m_renderer_surface_2D;
    std::set<std::shared_ptr<VFRendering::IsosurfaceRenderer>> m_renderers_isosurface;

    void setupRenderers();
    bool m_gl_initialized;
    bool m_suspended;

    int n_cell_step;
    int n_basis_atoms;

    const VFRendering::Options & options() const;

    // Parameters
    Colormap m_colormap_general;
    Colormap m_colormap_arrows;
    int m_colormap_rotation;
    bool m_colormap_invert_z;
    bool m_colormap_invert_xy;
    glm::vec3 m_colormap_cardinal_a;
    glm::vec3 m_colormap_cardinal_b;
    glm::vec3 m_colormap_cardinal_c;
    glm::vec2 m_x_range_direction;
    glm::vec2 m_y_range_direction;
    glm::vec2 m_z_range_direction;
    glm::vec2 m_x_range_position;
    glm::vec2 m_y_range_position;
    glm::vec2 m_z_range_position;
    int m_cell_a_min, m_cell_a_max, m_cell_b_min, m_cell_b_max, m_cell_c_min, m_cell_c_max;
    glm::vec2 m_surface_x_range;
    glm::vec2 m_surface_y_range;
    glm::vec2 m_surface_z_range;

    // Visualisation
    VFRendering::View m_view;
    VFRendering::VectorField m_vf;
    VFRendering::VectorField m_vf_surf2D;

    // Interaction mode
    InteractionMode m_interactionmode;
    bool regular_mode_perspective;
    glm::vec3 regular_mode_cam_pos;
    glm::vec3 regular_mode_cam_focus;
    glm::vec3 regular_mode_cam_up;
    // Calculate coordinates relative to the system center from QT device pixel coordinates
    //  This assumes that mouse_pos is relative to the top left corner of the widget.
    //  winsize should be the device pixel size of the widget.
    //  This function also assumes an orthogonal z-projection.
    glm::vec2 system_coords_from_mouse( glm::vec2 mouse_pos, glm::vec2 winsize );
    float system_radius_from_relative( float radius, glm::vec2 winsize );
    QTimer * m_timer_drag;
    QTimer * m_timer_drag_decoration;
    void dragpaste();
    void defectpaste();
    void pinningpaste();
    bool m_dragging;
    glm::vec2 last_drag_coords;

    // mouse decoration
    void updateMouseDecoration();
    MouseDecoratorWidget * mouse_decoration;
    float drag_radius;

    // Persistent Settings
    void writeSettings();
    void readSettings();
};

#endif
