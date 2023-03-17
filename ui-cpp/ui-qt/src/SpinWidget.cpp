#include "SpinWidget.hpp"

#include <QMouseEvent>
#include <QTimer>
#include <QtWidgets>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/VectorSphereRenderer.hxx>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <algorithm>
#include <sstream>

SpinWidget::SpinWidget( std::shared_ptr<State> state, QWidget * parent )
        : QOpenGLWidget( parent ), m_vf( {}, {} ), m_vf_surf2D( {}, {} )
{
    this->state            = state;
    this->m_gl_initialized = false;
    this->m_suspended      = false;
    this->paste_atom_type  = 0;

    // QT Widget Settings
    setFocusPolicy( Qt::StrongFocus );
    QSizePolicy sizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding );
    sizePolicy.setHorizontalStretch( 0 );
    sizePolicy.setVerticalStretch( 0 );
    this->setSizePolicy( sizePolicy );
    this->setMinimumSize( 200, 200 );
    this->setBaseSize( 600, 600 );

    // Default VFRendering Settings

    setColormapGeneral( Colormap::HSV );
    setColormapArrows( Colormap::HSV );
    setColormapRotationInverted( 0, false, false, glm::vec3{ 1, 0, 0 }, glm::vec3{ 0, 1, 0 }, glm::vec3{ 0, 0, 1 } );

    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( 0.125f );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( 0.3f );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( 0.0625f );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( 0.35f );

    setOverallDirectionRange( { -1, 1 }, { -1, 1 }, { -1, 1 } );

    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    setOverallPositionRange( x_range, y_range, z_range );
    this->m_surface_x_range = x_range;
    this->m_surface_y_range = y_range;
    this->m_surface_z_range = z_range;

    int n_cell[3];
    Geometry_Get_N_Cells( state.get(), n_cell );
    this->m_cell_a_min = 0;
    this->m_cell_a_max = n_cell[0] - 1;
    this->m_cell_b_min = 0;
    this->m_cell_b_max = n_cell[1] - 1;
    this->m_cell_c_min = 0;
    this->m_cell_c_max = n_cell[2] - 1;

    this->m_source                    = 0;
    this->visMode                     = VisualizationMode::SYSTEM;
    this->m_location_coordinatesystem = WidgetLocation::BOTTOM_RIGHT;
    this->m_location_miniview         = WidgetLocation::BOTTOM_LEFT;
    this->show_arrows                 = true;
    this->show_boundingbox            = true;
    this->show_isosurface             = false;

    idx_cycle          = 0;
    slab_displacements = glm::vec3{ 0, 0, 0 };

    this->n_cell_step = 1;

    this->show_surface          = false;
    this->show_miniview         = true;
    this->show_coordinatesystem = true;

    //         Initial camera position
    this->_reset_camera                   = false;
    this->m_camera_rotate_free            = false;
    this->m_camera_projection_perspective = true;

    //        Initial drag mode settings
    drag_radius            = 80;
    this->mouse_decoration = new MouseDecoratorWidget( drag_radius );
    this->mouse_decoration->setMinimumSize( 2 * drag_radius, 2 * drag_radius );
    this->mouse_decoration->setMaximumSize( 2 * drag_radius, 2 * drag_radius );
    this->mouse_decoration->setParent( this );
    this->m_interactionmode       = InteractionMode::REGULAR;
    this->m_timer_drag            = new QTimer( this );
    this->m_timer_drag_decoration = new QTimer( this );
    this->m_dragging              = false;

    //         Setup Arrays
    this->updateData();

    //         Read persistent settings
    this->readSettings();
    this->show_arrows      = this->user_show_arrows;
    this->show_surface     = this->user_show_surface;
    this->show_isosurface  = this->user_show_isosurface;
    this->show_boundingbox = this->user_show_boundingbox;
}

void SpinWidget::setSuspended( bool suspended )
{
    this->m_suspended = suspended;
    if( !suspended )
        QTimer::singleShot( 1, this, SLOT( update() ) );
}

const VFRendering::View * SpinWidget::view()
{
    return &( this->m_view );
}

const VFRendering::VectorField * SpinWidget::vectorfield()
{
    return &( this->m_vf );
}

void SpinWidget::addIsosurface( std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer )
{
    this->m_renderers_isosurface.insert( renderer );
    if( m_gl_initialized )
        this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );
}

void SpinWidget::removeIsosurface( std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer )
{
    this->m_renderers_isosurface.erase( renderer );
    if( m_gl_initialized )
        this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );
}

// Return the relative mouse position [-1,1]
glm::vec2 relative_coords_from_mouse( glm::vec2 mouse_pos, glm::vec2 winsize )
{
    glm::vec2 relative = 2.0f * ( mouse_pos - 0.5f * winsize );
    relative.x /= winsize.x;
    relative.y /= winsize.y;
    return relative;
}

glm::vec2 SpinWidget::system_coords_from_mouse( glm::vec2 mouse_pos, glm::vec2 winsize )
{
    auto relative = relative_coords_from_mouse( mouse_pos, winsize );
    glm::vec4 proj_back{ relative.x, relative.y, 0, 0 };

    auto matrices   = VFRendering::Utilities::getMatrices( options(), winsize.x / winsize.y );
    auto model_view = glm::inverse( matrices.first );
    auto projection = glm::inverse( matrices.second );

    proj_back = proj_back * projection;
    proj_back = proj_back * model_view;

    auto camera_position = options().get<VFRendering::View::Option::CAMERA_POSITION>();

    return glm::vec2{ proj_back.x + camera_position.x, -proj_back.y + camera_position.y };
}

float SpinWidget::system_radius_from_relative( float radius, glm::vec2 winsize )
{
    auto r1 = system_coords_from_mouse( { 0.0f, 0.0f }, winsize );
    auto r2 = system_coords_from_mouse( { radius - 5, 0.0f }, winsize );
    return r2.x - r1.x;
}

void SpinWidget::dragpaste()
{
    QPoint localCursorPos = this->mapFromGlobal( cursor().pos() );
    QSize widgetSize      = this->size();

    glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
    glm::vec2 size{ widgetSize.width(), widgetSize.height() };

    glm::vec2 coords = system_coords_from_mouse( mouse_pos, size );
    float radius     = system_radius_from_relative( this->drag_radius, size );
    float rect[3]{ -1, -1, -1 };

    float current_position[3]{ coords.x, coords.y, 0.0f };
    float shift[3]{ last_drag_coords.x - coords.x, last_drag_coords.y - coords.y, 0.0f };
    Configuration_From_Clipboard_Shift( state.get(), shift, current_position, rect, radius );
}

void SpinWidget::defectpaste()
{
    QPoint localCursorPos = this->mapFromGlobal( cursor().pos() );
    QSize widgetSize      = this->size();

    glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
    glm::vec2 size{ widgetSize.width(), widgetSize.height() };

    glm::vec2 coords = system_coords_from_mouse( mouse_pos, size );
    float radius     = system_radius_from_relative( this->drag_radius, size );
    float rect[3]{ -1, -1, -1 };

    float current_position[3]{ coords.x, coords.y, 0.0f };
    float center[3];
    Geometry_Get_Center( this->state.get(), center );
    current_position[0] -= center[0];
    current_position[1] -= center[1];

    Configuration_Set_Atom_Type( state.get(), this->paste_atom_type, current_position, rect, radius );
}

void SpinWidget::pinningpaste()
{
    QPoint localCursorPos = this->mapFromGlobal( cursor().pos() );
    QSize widgetSize      = this->size();

    glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
    glm::vec2 size{ widgetSize.width(), widgetSize.height() };

    glm::vec2 coords = system_coords_from_mouse( mouse_pos, size );
    float radius     = system_radius_from_relative( this->drag_radius, size );
    float rect[3]{ -1, -1, -1 };

    float current_position[3]{ coords.x, coords.y, 0.0f };
    float center[3];
    Geometry_Get_Center( this->state.get(), center );
    current_position[0] -= center[0];
    current_position[1] -= center[1];

    Configuration_Set_Pinned( state.get(), this->m_dragging, current_position, rect, radius );
}

void SpinWidget::setPasteAtomType( int type )
{
    this->paste_atom_type = type;
}

void SpinWidget::initializeGL()
{
    if( m_interactionmode == InteractionMode::DRAG )
    {
        this->setCursor( Qt::BlankCursor );
    }
    else
    {
        mouse_decoration->hide();
    }

    // Initialize VectorField data
    this->updateVectorFieldGeometry();
    this->updateVectorFieldDirections();

    // Get GL context
    makeCurrent();
    // Initialize the visualisation options

    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    glm::vec3 bounding_box_center = { ( bounds_min[0] + bounds_max[0] ) / 2, ( bounds_min[1] + bounds_max[1] ) / 2,
                                      ( bounds_min[2] + bounds_max[2] ) / 2 };
    glm::vec3 bounding_box_side_lengths
        = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

    // Create renderers
    //    System
    this->m_renderer_arrows = std::make_shared<VFRendering::ArrowRenderer>( m_view, m_vf );

    float indi_length            = glm::length( bounds_max - bounds_min ) * 0.05;
    int indi_dashes              = 5;
    float indi_dashes_per_length = (float)indi_dashes / indi_length;

    bool periodical[3];
    Hamiltonian_Get_Boundary_Conditions( this->state.get(), periodical );
    glm::vec3 indis{ indi_length * periodical[0], indi_length * periodical[1], indi_length * periodical[2] };

    this->m_renderer_boundingbox
        = std::make_shared<VFRendering::BoundingBoxRenderer>( VFRendering::BoundingBoxRenderer::forCuboid(
            m_view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length ) );

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = { m_renderer_arrows, m_renderer_boundingbox };

    if( Geometry_Get_Dimensionality( this->state.get() ) == 2 )
    {
        // 2D Surface options
        // No options yet...
        this->m_renderer_surface_2D = std::make_shared<VFRendering::SurfaceRenderer>( m_view, m_vf_surf2D );
        this->m_renderer_surface    = m_renderer_surface_2D;
    }
    else if( Geometry_Get_Dimensionality( this->state.get() ) == 3 )
    {
        // 3D Surface options
        this->m_renderer_surface_3D = std::make_shared<VFRendering::IsosurfaceRenderer>( m_view, m_vf );
        this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( 0.0 );
        auto mini_diff = glm::vec2{ 0.00001f, -0.00001f };
        setSurface( x_range + mini_diff, y_range + mini_diff, z_range + mini_diff );

        this->m_renderer_surface = m_renderer_surface_3D;
    }

    this->m_system = std::make_shared<VFRendering::CombinedRenderer>( m_view, renderers );

    //    Sphere
    this->m_sphere = std::make_shared<VFRendering::VectorSphereRenderer>( m_view, m_vf );

    //    Coordinate cross
    this->m_coordinatesystem = std::make_shared<VFRendering::CoordinateSystemRenderer>( m_view );
    this->m_coordinatesystem->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>( true );

    // Setup the View
    this->setVisualizationMode( this->visMode );

    // Configure System (Setup the renderers)
    this->setSystemCycle( SystemMode( this->idx_cycle ) );
    this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );

    // Set renderers' colormaps
    this->setColormapArrows( this->colormap_arrows() );

    this->m_gl_initialized = true;
}

void SpinWidget::teardownGL()
{
    // GLSpins::terminate();
}

void SpinWidget::resizeGL( int width, int height )
{
    this->m_view.setFramebufferSize( width * devicePixelRatio(), height * devicePixelRatio() );
    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::screenShot( std::string filename )
{
    auto pixmap = this->grab();
    pixmap.save( ( filename + ".png" ).c_str() );
}

void SpinWidget::updateVectorFieldGeometry()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, int( ceil( ( m_cell_a_max - m_cell_a_min + 1.0 ) / n_cell_step ) ) ),
                            std::max( 1, int( ceil( ( m_cell_b_max - m_cell_b_min + 1.0 ) / n_cell_step ) ) ),
                            std::max( 1, int( ceil( ( m_cell_c_max - m_cell_c_min + 1.0 ) / n_cell_step ) ) ) };

    int nos_draw = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Positions of the vectorfield
    std::vector<glm::vec3> positions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Positions
    //        get pointer
    scalar * spin_pos;
    int * atom_types;
    spin_pos   = Geometry_Get_Positions( state.get() );
    atom_types = Geometry_Get_Atom_Types( state.get() );
    int icell  = 0;
    for( int cell_c = m_cell_c_min; cell_c < m_cell_c_max + 1; cell_c += n_cell_step )
    {
        for( int cell_b = m_cell_b_min; cell_b < m_cell_b_max + 1; cell_b += n_cell_step )
        {
            for( int cell_a = m_cell_a_min; cell_a < m_cell_a_max + 1; cell_a += n_cell_step )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis + n_cell_atoms * ( cell_a + n_cells[0] * ( cell_b + n_cells[1] * cell_c ) );
                    positions[icell] = glm::vec3( spin_pos[3 * idx], spin_pos[1 + 3 * idx], spin_pos[2 + 3 * idx] );
                    ++icell;
                }
            }
        }
    }

    // Generate the right geometry (triangles and tetrahedra)
    VFRendering::Geometry geometry;
    VFRendering::Geometry geometry_surf2D;
    //      get tetrahedra
    if( Geometry_Get_Dimensionality( state.get() ) == 3 )
    {
        if( ( n_cells_draw[0] <= 1 || n_cells_draw[1] <= 1 || n_cells_draw[2] <= 1 ) )
        {
            geometry = VFRendering::Geometry( positions, {}, {}, true );
        }
        else
        {
            int temp_ranges[6]
                = { m_cell_a_min, m_cell_a_max + 1, m_cell_b_min, m_cell_b_max + 1, m_cell_c_min, m_cell_c_max + 1 };
            const std::array<VFRendering::Geometry::index_type, 4> * tetrahedra_indices_ptr = nullptr;
            int num_tetrahedra = Geometry_Get_Tetrahedra_Ranged(
                state.get(), reinterpret_cast<const int **>( &tetrahedra_indices_ptr ), n_cell_step, temp_ranges );
            std::vector<std::array<VFRendering::Geometry::index_type, 4>> tetrahedra_indices(
                tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra );
            geometry = VFRendering::Geometry( positions, {}, tetrahedra_indices, false );
        }
    }
    else if( Geometry_Get_Dimensionality( state.get() ) == 2 )
    {
        // Determine two basis vectors
        std::array<glm::vec3, 2> basis;
        float eps = 1e-6;
        for( int i = 1, j = 0; i < nos_draw && j < 2; ++i )
        {
            if( glm::length( positions[i] - positions[0] ) > eps )
            {
                if( j < 1 )
                {
                    basis[j] = glm::normalize( positions[i] - positions[0] );
                    ++j;
                }
                else
                {
                    if( 1 - std::abs( glm::dot( basis[0], glm::normalize( positions[i] - positions[0] ) ) ) > eps )
                    {
                        basis[j] = glm::normalize( positions[i] - positions[0] );
                        ++j;
                    }
                }
            }
        }

        int n_cells[3];
        Geometry_Get_N_Cells( this->state.get(), n_cells );
        float bounds_min[3], bounds_max[3];
        Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
        float density = 0.01f;
        if( n_cells[0] > 1 )
            density = std::max( density, n_cells[0] / ( bounds_max[0] - bounds_min[0] ) );
        if( n_cells[1] > 1 )
            density = std::max( density, n_cells[1] / ( bounds_max[1] - bounds_min[1] ) );
        if( n_cells[2] > 1 )
            density = std::max( density, n_cells[2] / ( bounds_max[2] - bounds_min[2] ) );
        density /= n_cell_step;
        glm::vec3 normal = this->arrowSize() / density * glm::normalize( glm::cross( basis[0], basis[1] ) );

        // By default, +z is up, which is where we want the normal oriented towards
        if( glm::dot( normal, glm::vec3{ 0, 0, 1 } ) < 1e-6 )
            normal = -normal;

        // Rectilinear with one basis atom
        if( n_cell_atoms == 1 && std::abs( glm::dot( basis[0], basis[1] ) ) < 1e-6 && glm::length( basis[0] ) > 1e-6
            && glm::length( basis[1] )
                   > 1e-6 ) // Check for length of the basis so that we do not get 1D geometries here
        {
            std::vector<float> xs( n_cells_draw[0] ), ys( n_cells_draw[1] ), zs( n_cells_draw[2] );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = positions[i].x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = positions[i * n_cells_draw[0]].y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = positions[i * n_cells_draw[0] * n_cells_draw[1]].z;
            geometry = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = ( positions[i] - normal ).x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = ( positions[i * n_cells_draw[0]] - normal ).y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = ( positions[i * n_cells_draw[0] * n_cells_draw[1]] - normal ).z;
            geometry_surf2D = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
        }
        // All others
        else
        {
            const std::array<VFRendering::Geometry::index_type, 3> * triangle_indices_ptr = nullptr;
            // int num_triangles = Geometry_Get_Triangulation(state.get(), reinterpret_cast<const int
            // **>(&triangle_indices_ptr), n_cell_step);
            int temp_ranges[6]
                = { m_cell_a_min, m_cell_a_max + 1, m_cell_b_min, m_cell_b_max + 1, m_cell_c_min, m_cell_c_max + 1 };
            int num_triangles = Geometry_Get_Triangulation_Ranged(
                state.get(), reinterpret_cast<const int **>( &triangle_indices_ptr ), n_cell_step, temp_ranges, -1,
                -1 );

            // If the geometry cannot be triangulated, it may e.g be one dimensional due to filters etc. we push a dummy
            // triangle, otherwise we get a lot of VFRendering error messages
            if( num_triangles < 1 && this->show_surface )
            {
                num_triangles        = 1;
                triangle_indices_ptr = new const std::array<VFRendering::Geometry::index_type, 3>();
            }

            std::vector<std::array<VFRendering::Geometry::index_type, 3>> triangle_indices(
                triangle_indices_ptr, triangle_indices_ptr + num_triangles );
            geometry = VFRendering::Geometry( positions, triangle_indices, {}, true );
            for( int i = 0; i < nos_draw; ++i )
                positions[i] = positions[i] - normal;
            geometry_surf2D = VFRendering::Geometry( positions, triangle_indices, {}, true );
        }

        // Update the vectorfield geometry
        this->m_vf_surf2D.updateGeometry( geometry_surf2D );
    }
    else
    {
        geometry = VFRendering::Geometry( positions, {}, {}, true );
    }

    // Update the vectorfield
    this->m_vf.updateGeometry( geometry );
}

void SpinWidget::updateVectorFieldDirections()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, int( ceil( ( m_cell_a_max - m_cell_a_min + 1.0 ) / n_cell_step ) ) ),
                            std::max( 1, int( ceil( ( m_cell_b_max - m_cell_b_min + 1.0 ) / n_cell_step ) ) ),
                            std::max( 1, int( ceil( ( m_cell_c_max - m_cell_c_min + 1.0 ) / n_cell_step ) ) ) };

    int nos_draw = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Directions of the vectorfield
    std::vector<glm::vec3> directions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Directions
    //        get pointer
    scalar * spins;
    int * atom_types;
    atom_types = Geometry_Get_Atom_Types( state.get() );
    if( this->m_source == 0 )
        spins = System_Get_Spin_Directions( state.get() );
    else if( this->m_source == 1 )
        spins = System_Get_Effective_Field( state.get() );
    else
        spins = System_Get_Spin_Directions( state.get() );
    //        copy
    /*positions.assign(spin_pos, spin_pos + 3*nos);
    directions.assign(spins, spins + 3*nos);*/
    int icell = 0;
    for( int cell_c = m_cell_c_min; cell_c < m_cell_c_max + 1; cell_c += n_cell_step )
    {
        for( int cell_b = m_cell_b_min; cell_b < m_cell_b_max + 1; cell_b += n_cell_step )
        {
            for( int cell_a = m_cell_a_min; cell_a < m_cell_a_max + 1; cell_a += n_cell_step )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis + n_cell_atoms * ( cell_a + n_cells[0] * ( cell_b + n_cells[1] * cell_c ) );
                    directions[icell] = glm::vec3( spins[3 * idx], spins[1 + 3 * idx], spins[2 + 3 * idx] );
                    if( atom_types[idx] < 0 )
                        directions[icell] *= 0;
                    ++icell;
                }
            }
        }
    }
    //        rescale if effective field
    if( this->m_source == 1 )
    {
        float max_length = 0;
        for( auto direction : directions )
        {
            max_length = std::max( max_length, glm::length( direction ) );
        }
        if( max_length > 0 )
        {
            for( auto & direction : directions )
            {
                direction /= max_length;
            }
        }
    }

    // Update the vectorfield
    this->m_vf.updateVectors( directions );

    if( Geometry_Get_Dimensionality( state.get() ) == 2 )
        this->m_vf_surf2D.updateVectors( directions );
}

void SpinWidget::updateData( bool update_directions, bool update_geometry, bool update_camera )
{
    // Update the VectorField
    if( update_directions )
        this->updateVectorFieldDirections();

    if( update_geometry )
        this->updateVectorFieldGeometry();

    // Update the View
    if( update_camera )
    {
        float b_min[3], b_max[3];
        Geometry_Get_Bounds( state.get(), b_min, b_max );
        glm::vec3 bounds_min = glm::make_vec3( b_min );
        glm::vec3 bounds_max = glm::make_vec3( b_max );
        glm::vec3 center     = ( bounds_min + bounds_max ) * 0.5f;
        this->m_view.setOption<VFRendering::View::Option::SYSTEM_CENTER>( center );
        if( this->_reset_camera )
        {
            setCameraToDefault();
            this->_reset_camera = false;
        }
    }

    // Update Widget
    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::paintGL()
{
    if( this->m_suspended )
        return;

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() )
        || this->m_dragging )
    {
        this->updateData( true, false, true );
    }

    this->m_view.draw();
}

void SpinWidget::setVisualisationSource( int source )
{
    this->m_source = source;
}

void SpinWidget::mousePressEvent( QMouseEvent * event )
{
    if( this->m_suspended )
        return;

    m_previous_mouse_position = event->pos();

    if( m_interactionmode == InteractionMode::DRAG || m_interactionmode == InteractionMode::DEFECT
        || m_interactionmode == InteractionMode::PIN )
    {
        if( event->button() == Qt::LeftButton )
        {
            QPoint localCursorPos = this->mapFromGlobal( cursor().pos() );
            QSize widgetSize      = this->size();
            glm::vec2 mouse_pos{ localCursorPos.x(), localCursorPos.y() };
            glm::vec2 size{ widgetSize.width(), widgetSize.height() };
            last_drag_coords = system_coords_from_mouse( mouse_pos, size );

            m_timer_drag->stop();
            // Copy spin configuration
            Configuration_To_Clipboard( state.get() );
            // Set up Update Timers
            if( m_interactionmode == InteractionMode::DRAG )
                connect( m_timer_drag, &QTimer::timeout, this, &SpinWidget::dragpaste );
            else if( m_interactionmode == InteractionMode::DEFECT )
                connect( m_timer_drag, &QTimer::timeout, this, &SpinWidget::defectpaste );
            else if( m_interactionmode == InteractionMode::PIN )
                connect( m_timer_drag, &QTimer::timeout, this, &SpinWidget::pinningpaste );
            float ips = Simulation_Get_IterationsPerSecond( state.get() );
            if( ips > 1000 )
            {
                m_timer_drag->start( 1 );
            }
            else if( ips > 0 )
            {
                m_timer_drag->start( (int)( 1000 / ips ) );
            }
            m_dragging = true;
        }
    }
}

void SpinWidget::mouseReleaseEvent( QMouseEvent * event )
{
    if( this->m_suspended )
        return;

    if( m_interactionmode == InteractionMode::DRAG || m_interactionmode == InteractionMode::DEFECT
        || m_interactionmode == InteractionMode::PIN )
    {
        if( event->button() == Qt::LeftButton )
        {
            m_timer_drag->stop();
            m_dragging = false;
        }
        else if( event->button() == Qt::RightButton )
        {
            if( m_interactionmode == InteractionMode::DRAG )
                dragpaste();
            else if( m_interactionmode == InteractionMode::DEFECT )
                defectpaste();
            else if( m_interactionmode == InteractionMode::PIN )
                pinningpaste();
            this->updateData();
        }
    }
}

void SpinWidget::mouseMoveEvent( QMouseEvent * event )
{
    if( this->m_suspended )
        return;

    if( m_interactionmode == InteractionMode::DRAG )
    {
        dragpaste();
        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
    else if( m_interactionmode == InteractionMode::DEFECT )
    {
        defectpaste();
        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
    else if( m_interactionmode == InteractionMode::PIN )
    {
        pinningpaste();
        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
    else if( event->buttons() & Qt::LeftButton || event->buttons() & Qt::RightButton )
    {
        float scale = 1;
        if( event->modifiers() & Qt::ShiftModifier )
            scale = 0.1f;

        glm::vec2 current_mouse_position
            = glm::vec2( event->pos().x(), event->pos().y() ) * (float)devicePixelRatio() * scale;
        glm::vec2 previous_mouse_position = glm::vec2( m_previous_mouse_position.x(), m_previous_mouse_position.y() )
                                            * (float)devicePixelRatio() * scale;
        m_previous_mouse_position = event->pos();

        VFRendering::CameraMovementModes movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
        if( this->m_camera_rotate_free )
            movement_mode = VFRendering::CameraMovementModes::ROTATE_FREE;
        if( ( event->modifiers() & Qt::AltModifier ) == Qt::AltModifier || event->buttons() & Qt::RightButton )
        {
            movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
        }
        this->m_view.mouseMove( previous_mouse_position, current_mouse_position, movement_mode );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::wheelEvent( QWheelEvent * event )
{
    float scale = 1;

    if( event->modifiers() & Qt::ShiftModifier )
    {
        scale = 0.1f;
    }

    if( event->modifiers() & Qt::ControlModifier )
    {
        float wheel_delta = scale * event->angleDelta().y() / 10.0f;
        drag_radius       = std::max( 1.0f, std::min( 500.0f, drag_radius + wheel_delta ) );
        this->mouse_decoration->setRadius( drag_radius );
        this->mouse_decoration->setMinimumSize( 2 * drag_radius, 2 * drag_radius );
        this->mouse_decoration->setMaximumSize( 2 * drag_radius, 2 * drag_radius );
    }
    else
    {
        float wheel_delta = event->angleDelta().y();
        this->m_view.mouseScroll( -wheel_delta * 0.1 * scale );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::updateMouseDecoration()
{
    auto pos = this->mapFromGlobal( QCursor::pos() - QPoint( drag_radius, drag_radius ) );
    this->mouse_decoration->move( (int)pos.x(), (int)pos.y() );
}

float SpinWidget::getFramesPerSecond() const
{
    return this->m_view.getFramerate();
}

const VFRendering::Options & SpinWidget::options() const
{
    return this->m_view.options();
}

void SpinWidget::moveCamera( float backforth, float rightleft, float updown )
{
    if( this->m_suspended )
        return;

    auto movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
    this->m_view.mouseMove( { 0, 0 }, { rightleft, updown }, movement_mode );
    this->m_view.mouseScroll( backforth * 0.1 );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::rotateCamera( float theta, float phi )
{
    if( this->m_suspended )
        return;

    if( this->m_interactionmode == InteractionMode::DRAG )
    {
        theta = 0;
    }
    VFRendering::CameraMovementModes movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
    if( this->m_camera_rotate_free )
        movement_mode = VFRendering::CameraMovementModes::ROTATE_FREE;
    this->m_view.mouseMove( { 0, 0 }, { phi, theta }, movement_mode );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

//////////////////////////////////////////////////////////////////////////////////////
int SpinWidget::visualisationNCellSteps()
{
    return this->n_cell_step;
}

void SpinWidget::setVisualisationNCellSteps( int n_cell_steps )
{
    float size_before = this->arrowSize();
    this->n_cell_step = n_cell_steps;
    this->setArrows( size_before, this->arrowLOD() );
    this->updateData();
}

///// --- Mode ---
void SpinWidget::setVisualizationMode( SpinWidget::VisualizationMode visualization_mode )
{
    if( visualization_mode == SpinWidget::VisualizationMode::SYSTEM )
    {
        this->visMode    = VisualizationMode::SYSTEM;
        this->m_mainview = this->m_system;
        this->m_miniview = this->m_sphere;
    }
    else if( visualization_mode == SpinWidget::VisualizationMode::SPHERE )
    {
        this->visMode    = VisualizationMode::SPHERE;
        this->m_mainview = this->m_sphere;
        this->m_miniview = this->m_system;
    }

    this->setupRenderers();
}

SpinWidget::VisualizationMode SpinWidget::visualizationMode()
{
    return this->visMode;
}

void SpinWidget::setInteractionMode( InteractionMode mode )
{
    if( mode == InteractionMode::DRAG || mode == InteractionMode::DEFECT || mode == InteractionMode::PIN )
    {
        // Save latest regular mode settings
        this->regular_mode_perspective = this->cameraProjection();
        this->regular_mode_cam_pos     = this->getCameraPositon();
        this->regular_mode_cam_focus   = this->getCameraFocus();
        this->regular_mode_cam_up      = this->getCameraUpVector();
        // Set cursor
        this->setCursor( Qt::BlankCursor );
        if( mode == InteractionMode::DRAG )
            this->mouse_decoration->setColors( Qt::white, Qt::black );
        else if( mode == InteractionMode::DEFECT )
            this->mouse_decoration->setColors( Qt::white, Qt::darkRed );
        else if( mode == InteractionMode::PIN )
            this->mouse_decoration->setColors( Qt::white, Qt::darkBlue );
        this->mouse_decoration->show();
        // Apply camera changes
        this->setCameraToZ();
        this->setCameraProjection( false );
        // Set mode after changes so that changes are not blocked
        this->m_interactionmode = mode;
        // Set up update timers
        m_timer_drag_decoration->stop();
        connect( m_timer_drag_decoration, &QTimer::timeout, this, &SpinWidget::updateMouseDecoration );
        m_timer_drag_decoration->start( 5 );
    }
    else
    {
        // Stop update timers
        m_timer_drag_decoration->stop();
        // Unset cursor
        this->unsetCursor();
        this->mouse_decoration->hide();
        // Set mode before changes so that changes are not blocked
        this->m_interactionmode = mode;
        // Apply camera changes
        this->setCameraProjection( this->regular_mode_perspective );
        this->setCameraPosition( this->regular_mode_cam_pos );
        this->setCameraFocus( this->regular_mode_cam_focus );
        this->setCameraUpVector( this->regular_mode_cam_up );
    }

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

SpinWidget::InteractionMode SpinWidget::interactionMode()
{
    return this->m_interactionmode;
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- MiniView ---
void SpinWidget::setVisualizationMiniview( bool show, SpinWidget::WidgetLocation location )
{
    enableMiniview( show );
    setMiniviewPosition( location );
}

bool SpinWidget::isMiniviewEnabled() const
{
    return this->show_miniview;
}

void SpinWidget::enableMiniview( bool enabled )
{
    this->show_miniview = enabled;
    setupRenderers();
}

SpinWidget::WidgetLocation SpinWidget::miniviewPosition() const
{
    return this->m_location_miniview;
}

void SpinWidget::setMiniviewPosition( SpinWidget::WidgetLocation location )
{
    this->m_location_miniview = location;
    this->setupRenderers();
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Coordinate System ---
void SpinWidget::setVisualizationCoordinatesystem( bool show, SpinWidget::WidgetLocation location )
{
    enableCoordinateSystem( show );
    setCoordinateSystemPosition( location );
}

bool SpinWidget::isCoordinateSystemEnabled() const
{
    return this->show_coordinatesystem;
}

void SpinWidget::enableCoordinateSystem( bool enabled )
{
    this->show_coordinatesystem = enabled;
    setupRenderers();
}

SpinWidget::WidgetLocation SpinWidget::coordinateSystemPosition() const
{
    return this->m_location_coordinatesystem;
}

void SpinWidget::setCoordinateSystemPosition( SpinWidget::WidgetLocation location )
{
    this->m_location_coordinatesystem = location;
    this->setupRenderers();
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- System ---

void SpinWidget::setSlabRanges()
{
    float f_center[3], bounds_min[3], bounds_max[3];
    Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
    Geometry_Get_Center( state.get(), f_center );
    glm::vec2 x_range( bounds_min[0], bounds_max[0] );
    glm::vec2 y_range( bounds_min[1], bounds_max[1] );
    glm::vec2 z_range( bounds_min[2], bounds_max[2] );
    glm::vec3 center( f_center[0], f_center[1], f_center[2] );
    center += this->slab_displacements;

    float delta = 0.51f;

    switch( this->idx_cycle )
    {
        case 2:
        {
            if( (int)center.x == center.x )
            {
                center.x += 0.5;
            }
            x_range = { center[0] - delta, center[0] + delta };
            break;
        }
        case 3:
        {
            if( (int)center.y == center.y )
            {
                center.y += 0.5;
            }
            y_range = { center[1] - delta, center[1] + delta };
            break;
        }
        case 4:
        {
            if( (int)center.z == center.z )
            {
                center.z += 0.5;
            }
            z_range = { center[2] - delta, center[2] + delta };
            break;
        }
    }

    float mini_shift = 1e-5f;
    x_range.x        = std::max( bounds_min[0] + mini_shift, x_range.x );
    x_range.y        = std::min( bounds_max[0] - mini_shift, x_range.y );
    y_range.x        = std::max( bounds_min[1] + mini_shift, y_range.x );
    y_range.y        = std::min( bounds_max[1] - mini_shift, y_range.y );
    z_range.x        = std::max( bounds_min[2] + mini_shift, z_range.x );
    z_range.y        = std::min( bounds_max[2] - mini_shift, z_range.y );

    this->setSurface( x_range, y_range, z_range );
}

void SpinWidget::setSystemCycle( SystemMode mode )
{
    this->idx_cycle = (int)mode;

    switch( mode )
    {
        case SystemMode::CUSTOM:
        {
            // User settings
            this->show_arrows      = this->user_show_arrows;
            this->show_surface     = this->user_show_surface;
            this->show_isosurface  = this->user_show_isosurface;
            this->show_boundingbox = this->user_show_boundingbox;
            this->setVerticalFieldOfView( this->user_fov );
            // Camera
            break;
        }
        case SystemMode::ISOSURFACE:
        {
            // Isosurface
            this->show_arrows     = false;
            this->show_surface    = false;
            this->show_isosurface = true;
            this->setVerticalFieldOfView( this->user_fov );
            break;
        }
        case SystemMode::SLAB_X:
        {
            // Slab x
            this->show_arrows     = false;
            this->show_surface    = true;
            this->show_isosurface = false;
            // camera
            // this->setCameraToX();
            // this->setVerticalFieldOfView(0);
            break;
        }
        case SystemMode::SLAB_Y:
        {
            // Slab y
            this->show_arrows     = false;
            this->show_surface    = true;
            this->show_isosurface = false;
            // camera
            // this->setCameraToY();
            // this->setVerticalFieldOfView(0);
            break;
        }
        case SystemMode::SLAB_Z:
        {
            // Slab z
            this->show_arrows     = false;
            this->show_surface    = true;
            this->show_isosurface = false;
            // camera
            // this->setCameraToZ();
            // this->setVerticalFieldOfView(0);
            break;
        }
    }
    this->setSlabRanges();
}

void SpinWidget::cycleSystem( SystemMode mode )
{
    // save possible user settings
    if( this->idx_cycle == 0 )
    {
        this->user_show_arrows      = this->show_arrows;
        this->user_show_surface     = this->show_surface;
        this->user_show_isosurface  = this->show_isosurface;
        this->user_show_boundingbox = this->show_boundingbox;
        this->user_fov              = this->verticalFieldOfView();
    }

    this->idx_cycle = (int)mode;

    this->setSystemCycle( mode );

    this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );
}

void SpinWidget::cycleSystem( bool forward )
{
    // save possible user settings
    if( this->idx_cycle == 0 )
    {
        this->user_show_arrows      = this->show_arrows;
        this->user_show_surface     = this->show_surface;
        this->user_show_isosurface  = this->show_isosurface;
        this->user_show_boundingbox = this->show_boundingbox;
        this->user_fov              = this->verticalFieldOfView();
    }

    if( forward )
    {
        ++this->idx_cycle;
    }
    else
    {
        --this->idx_cycle;
    }
    if( this->idx_cycle < 0 )
        idx_cycle += 5;
    this->idx_cycle = this->idx_cycle % 5;

    this->setSystemCycle( SystemMode( this->idx_cycle ) );

    this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );
}

SpinWidget::SystemMode SpinWidget::systemCycle()
{
    return SystemMode( this->idx_cycle );
}

glm::vec2 SpinWidget::surfaceXRange() const
{
    return m_surface_x_range;
}

glm::vec2 SpinWidget::surfaceYRange() const
{
    return m_surface_y_range;
}

glm::vec2 SpinWidget::surfaceZRange() const
{
    return m_surface_z_range;
}

/////    enable
void SpinWidget::enableSystem( bool arrows, bool boundingbox, bool surface, bool isosurface )
{
    this->show_arrows      = arrows;
    this->show_boundingbox = boundingbox;
    this->show_surface     = surface;
    this->show_isosurface  = isosurface;

    if( idx_cycle == 0 )
    {
        this->user_show_arrows      = this->show_arrows;
        this->user_show_surface     = this->show_surface;
        this->user_show_isosurface  = this->show_isosurface;
        this->user_show_boundingbox = this->show_boundingbox;
        this->user_fov              = this->verticalFieldOfView();
    }

    // Create System
    std::vector<std::shared_ptr<VFRendering::RendererBase>> system( 0 );
    if( show_arrows )
        system.push_back( this->m_renderer_arrows );
    if( show_boundingbox )
        system.push_back( this->m_renderer_boundingbox );
    if( show_surface
        && ( Geometry_Get_Dimensionality( this->state.get() ) == 2
             || Geometry_Get_Dimensionality( this->state.get() ) == 3 ) )
        system.push_back( this->m_renderer_surface );
    if( show_isosurface && ( Geometry_Get_Dimensionality( this->state.get() ) == 3 ) )
    {
        for( auto & iso : this->m_renderers_isosurface )
            system.push_back( iso );
    }
    this->m_system = std::make_shared<VFRendering::CombinedRenderer>( m_view, system );
    //*this->m_system = VFRendering::CombinedRenderer(m_view, system);

    if( this->visMode == VisualizationMode::SYSTEM )
        this->m_mainview = this->m_system;
    else
        this->m_miniview = this->m_system;

    this->setupRenderers();
}

void SpinWidget::moveSlab( int amount )
{
    float f_center[3], bounds_min[3], bounds_max[3];
    Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
    Geometry_Get_Center( state.get(), f_center );
    for( int i = 0; i < 3; ++i )
        if( (int)f_center[i] == f_center[i] )
            f_center[i] += 0.5;
    glm::vec3 center( f_center[0], f_center[1], f_center[2] );
    glm::vec3 pos = center + this->slab_displacements;

    float cell_bounds_min[3], cell_bounds_max[3];
    Geometry_Get_Cell_Bounds( state.get(), cell_bounds_min, cell_bounds_max );
    glm::vec3 cell_size{ cell_bounds_max[0] - cell_bounds_min[0], cell_bounds_max[1] - cell_bounds_min[1],
                         cell_bounds_max[2] - cell_bounds_min[2] };
    if( this->idx_cycle == 2 )
    {
        // X
        amount *= cell_size[0];
        this->slab_displacements[0] = std::min(
                                          std::max( bounds_min[0] + 0.5f * cell_size[0], pos[0] + amount ),
                                          bounds_max[0] - 0.5f * cell_size[0] )
                                      - center[0];
    }
    else if( this->idx_cycle == 3 )
    {
        // Y
        amount *= cell_size[1];
        this->slab_displacements[1] = std::min(
                                          std::max( bounds_min[1] + 0.5f * cell_size[1], pos[1] + amount ),
                                          bounds_max[1] - 0.5f * cell_size[1] )
                                      - center[1];
    }
    else if( this->idx_cycle == 4 )
    {
        // Z
        amount *= cell_size[2];
        this->slab_displacements[2] = std::min(
                                          std::max( bounds_min[2] + 0.5f * cell_size[2], pos[2] + amount ),
                                          bounds_max[2] - 0.5f * cell_size[2] )
                                      - center[2];
    }

    this->setSlabRanges();
}

/////    Arrows
void SpinWidget::setArrows( float size, int lod )
{
    if( lod < 3 )
        lod = 3;

    // defaults
    float coneradius     = 0.25f;
    float coneheight     = 0.6f;
    float cylinderradius = 0.125f;
    float cylinderheight = 0.7f;

    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    glm::vec3 bounding_box_center = { ( bounds_min[0] + bounds_max[0] ) / 2, ( bounds_min[1] + bounds_max[1] ) / 2,
                                      ( bounds_min[2] + bounds_max[2] ) / 2 };
    glm::vec3 bounding_box_side_lengths
        = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );

    float density = 0.01f;
    if( n_cells[0] > 1 )
        density = std::max( density, n_cells[0] / ( bounds_max[0] - bounds_min[0] ) );
    if( n_cells[1] > 1 )
        density = std::max( density, n_cells[1] / ( bounds_max[1] - bounds_min[1] ) );
    if( n_cells[2] > 1 )
        density = std::max( density, n_cells[2] / ( bounds_max[2] - bounds_min[2] ) );
    density /= n_cell_step;

    makeCurrent();
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( coneheight * size / density );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( coneradius * size / density );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( cylinderheight * size / density );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( cylinderradius * size / density );
    this->m_view.setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>( lod );

    this->updateVectorFieldGeometry();

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

float SpinWidget::arrowSize() const
{
    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    glm::vec3 bounding_box_center = { ( bounds_min[0] + bounds_max[0] ) / 2, ( bounds_min[1] + bounds_max[1] ) / 2,
                                      ( bounds_min[2] + bounds_max[2] ) / 2 };
    glm::vec3 bounding_box_side_lengths
        = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );

    float density = 0.01f;
    if( n_cells[0] > 1 )
        density = std::max( density, n_cells[0] / ( bounds_max[0] - bounds_min[0] ) );
    if( n_cells[1] > 1 )
        density = std::max( density, n_cells[1] / ( bounds_max[1] - bounds_min[1] ) );
    if( n_cells[2] > 1 )
        density = std::max( density, n_cells[2] / ( bounds_max[2] - bounds_min[2] ) );
    density /= n_cell_step;

    float size = options().get<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>() / 0.6f * density;
    return size;
}

int SpinWidget::arrowLOD() const
{
    int LOD = options().get<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>();
    return LOD;
}

/////    Overall Range Directions
glm::vec2 SpinWidget::xRangeDirection() const
{
    return m_x_range_direction;
}
glm::vec2 SpinWidget::yRangeDirection() const
{
    return m_y_range_direction;
}
glm::vec2 SpinWidget::zRangeDirection() const
{
    return m_z_range_direction;
}

void SpinWidget::setOverallDirectionRange( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range )
{
    m_x_range_direction = x_range;
    m_y_range_direction = y_range;
    m_z_range_direction = z_range;

    this->updateIsVisibleImplementation();
}

/////    Overall Range Position
glm::vec2 SpinWidget::xRangePosition() const
{
    return m_x_range_position;
}
glm::vec2 SpinWidget::yRangePosition() const
{
    return m_y_range_position;
}
glm::vec2 SpinWidget::zRangePosition() const
{
    return m_z_range_position;
}

void SpinWidget::setOverallPositionRange( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range )
{
    m_x_range_position = x_range;
    m_y_range_position = y_range;
    m_z_range_position = z_range;

    this->updateIsVisibleImplementation();
}

void SpinWidget::updateIsVisibleImplementation()
{
    float epsilon = 1e-5;
    std::ostringstream sstream;
    std::string is_visible_implementation;
    sstream << "bool is_visible(vec3 position, vec3 direction) {";
    //        position
    // X
    if( m_x_range_position.x >= m_x_range_position.y )
    {
        sstream << "bool is_visible_x_pos = true;";
    }
    else
    {
        sstream << "float x_min_pos = ";
        sstream << m_x_range_position.x - epsilon;
        sstream << "; float x_max_pos = ";
        sstream << m_x_range_position.y + epsilon;
        sstream << "; bool is_visible_x_pos = position.x <= x_max_pos && position.x >= x_min_pos;";
    }
    // Y
    if( m_y_range_position.x >= m_y_range_position.y )
    {
        sstream << "bool is_visible_y_pos = true;";
    }
    else
    {
        sstream << "float y_min_pos = ";
        sstream << m_y_range_position.x - epsilon;
        sstream << "; float y_max_pos = ";
        sstream << m_y_range_position.y + epsilon;
        sstream << "; bool is_visible_y_pos = position.y <= y_max_pos && position.y >= y_min_pos;";
    }
    // Z
    if( m_z_range_position.x >= m_z_range_position.y )
    {
        sstream << "bool is_visible_z_pos = true;";
    }
    else
    {
        sstream << "float z_min_pos = ";
        sstream << m_z_range_position.x - epsilon;
        sstream << "; float z_max_pos = ";
        sstream << m_z_range_position.y + epsilon;
        sstream << "; bool is_visible_z_pos = position.z <= z_max_pos && position.z >= z_min_pos;";
    }
    //        direction
    // X
    if( m_x_range_direction.x <= -1 && m_x_range_direction.y >= 1 )
    {
        sstream << "bool is_visible_x_dir = true;";
    }
    else if( m_x_range_direction.x <= -1 )
    {
        sstream << "float x_max_dir = ";
        sstream << m_x_range_direction.y + epsilon;
        sstream << "; bool is_visible_x_dir = normalize(direction).x <= x_max_dir;";
    }
    else if( m_x_range_direction.y >= 1 )
    {
        sstream << "float x_min_dir = ";
        sstream << m_x_range_direction.x - epsilon;
        sstream << "; bool is_visible_x_dir = normalize(direction).x >= x_min_dir;";
    }
    else
    {
        sstream << "float x_min_dir = ";
        sstream << m_x_range_direction.x - epsilon;
        sstream << "; float x_max_dir = ";
        sstream << m_x_range_direction.y + epsilon;
        sstream << "; float x_dir = normalize(direction).x; bool is_visible_x_dir = x_dir >= x_min_dir && x_dir <= "
                   "x_max_dir;";
    }
    // Y
    if( m_y_range_direction.x <= -1 && m_y_range_direction.y >= 1 )
    {
        sstream << "bool is_visible_y_dir = true;";
    }
    else if( m_y_range_direction.x <= -1 )
    {
        sstream << "float y_max_dir = ";
        sstream << m_y_range_direction.y + epsilon;
        sstream << "; bool is_visible_y_dir = normalize(direction).y <= y_max_dir;";
    }
    else if( m_y_range_direction.y >= 1 )
    {
        sstream << "float y_min_dir = ";
        sstream << m_y_range_direction.x - epsilon;
        sstream << "; bool is_visible_y_dir = normalize(direction).y >= y_min_dir;";
    }
    else
    {
        sstream << "float y_min_dir = ";
        sstream << m_y_range_direction.x - epsilon;
        sstream << "; float y_max_dir = ";
        sstream << m_y_range_direction.y + epsilon;
        sstream << "; float y_dir = normalize(direction).y;  bool is_visible_y_dir = y_dir >= y_min_dir && y_dir <= "
                   "y_max_dir;";
    }
    // Z
    if( m_z_range_direction.x <= -1 && m_z_range_direction.y >= 1 )
    {
        sstream << "bool is_visible_z_dir = true;";
    }
    else if( m_z_range_direction.x <= -1 )
    {
        sstream << "float z_max_dir = ";
        sstream << m_z_range_direction.y + epsilon;
        sstream << "; bool is_visible_z_dir = normalize(direction).z <= z_max_dir;";
    }
    else if( m_z_range_direction.y >= 1 )
    {
        sstream << "float z_min_dir = ";
        sstream << m_z_range_direction.x - epsilon;
        sstream << "; bool is_visible_z_dir = normalize(direction).z >= z_min_dir;";
    }
    else
    {
        sstream << "float z_min_dir = ";
        sstream << m_z_range_direction.x - epsilon;
        sstream << "; float z_max_dir = ";
        sstream << m_z_range_direction.y + epsilon;
        sstream << "; float z_dir = normalize(direction).z;  bool is_visible_z_dir = z_dir >= z_min_dir && z_dir <= "
                   "z_max_dir;";
    }
    //
    sstream << " return is_visible_x_pos && is_visible_y_pos && is_visible_z_pos && "
               "is_visible_x_dir && is_visible_y_dir && is_visible_z_dir; }";
    is_visible_implementation = sstream.str();
    makeCurrent();
    this->m_view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible_implementation );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

/////   Surface
void SpinWidget::setSurface( glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range )
{
    this->m_surface_x_range = x_range;
    this->m_surface_y_range = y_range;
    this->m_surface_z_range = z_range;

    makeCurrent();
    if( Geometry_Get_Dimensionality( this->state.get() ) == 2 )
    {
        // 2D Surface options
        // No options, yet...
    }
    else if( Geometry_Get_Dimensionality( this->state.get() ) == 3 )
    {
        // 3D Surface options
        if( ( x_range.x >= x_range.y ) || ( y_range.x >= y_range.y ) || ( z_range.x >= z_range.y ) )
        {
            this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
                [x_range, y_range, z_range]( const glm::vec3 & position, const glm::vec3 & direction )
                    -> VFRendering::IsosurfaceRenderer::isovalue_type
                {
                    /* The selected cuboid does not exist */
                    return 1;
                } );
        }
        else
        {
            this->m_renderer_surface_3D->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
                [x_range, y_range, z_range]( const glm::vec3 & position, const glm::vec3 & direction )
                    -> VFRendering::IsosurfaceRenderer::isovalue_type
                {
                    (void)direction;

                    /* Transform position in selected cuboid to position in unit cube
                     * [-1,1]^3 */
                    glm::vec3 min                 = { x_range.x, y_range.x, z_range.x };
                    glm::vec3 max                 = { x_range.y, y_range.y, z_range.y };
                    glm::vec3 normalized_position = 2.0f * ( position - min ) / ( max - min ) - 1.0f;

                    /* Calculate maximum metric / Chebyshev distance */
                    glm::vec3 absolute_normalized_position = glm::abs( normalized_position );
                    float max_norm                         = glm::max(
                        glm::max( absolute_normalized_position.x, absolute_normalized_position.y ),
                        absolute_normalized_position.z );

                    /* Translate so that the selected cuboid surface has an isovalue of 0 */
                    return max_norm - 1.0f;
                } );
        }
    }

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Sphere ---
glm::vec2 SpinWidget::spherePointSizeRange() const
{
    return options().get<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>();
}

void SpinWidget::setSpherePointSizeRange( glm::vec2 sphere_point_size_range )
{
    makeCurrent();
    this->m_view.setOption<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>( sphere_point_size_range );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Renderer Setup ---
void SpinWidget::setupRenderers()
{
    makeCurrent();

    // Get positions
    std::array<float, 4> position_miniview;
    if( this->m_location_miniview == SpinWidget::WidgetLocation::BOTTOM_LEFT )
        position_miniview = { 0, 0, 0.2f, 0.2f };
    else if( this->m_location_miniview == SpinWidget::WidgetLocation::BOTTOM_RIGHT )
        position_miniview = { 0.8f, 0, 0.2f, 0.2f };
    else if( this->m_location_miniview == SpinWidget::WidgetLocation::TOP_LEFT )
        position_miniview = { 0, 0.8f, 0.2f, 0.2f };
    else if( this->m_location_miniview == SpinWidget::WidgetLocation::TOP_RIGHT )
        position_miniview = { 0.8f, 0.8f, 0.2f, 0.2f };

    std::array<float, 4> position_coordinatesystem;
    if( this->m_location_coordinatesystem == SpinWidget::WidgetLocation::BOTTOM_LEFT )
        position_coordinatesystem = { 0, 0, 0.2f, 0.2f };
    else if( this->m_location_coordinatesystem == SpinWidget::WidgetLocation::BOTTOM_RIGHT )
        position_coordinatesystem = { 0.8f, 0, 0.2f, 0.2f };
    else if( this->m_location_coordinatesystem == SpinWidget::WidgetLocation::TOP_LEFT )
        position_coordinatesystem = { 0, 0.8f, 0.2f, 0.2f };
    else if( this->m_location_coordinatesystem == SpinWidget::WidgetLocation::TOP_RIGHT )
        position_coordinatesystem = { 0.8f, 0.8f, 0.2f, 0.2f };

    // Create renderers vector
    std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> renderers;
    renderers.push_back( { this->m_mainview, { 0, 0, 1, 1 } } );
    if( show_miniview )
        renderers.push_back( { this->m_miniview, position_miniview } );
    if( show_coordinatesystem )
        renderers.push_back( { this->m_coordinatesystem, position_coordinatesystem } );

    // Update View
    this->m_view.renderers( renderers, false );
    this->m_view.setOption<VFRendering::View::CAMERA_POSITION>( options().get<VFRendering::View::CAMERA_POSITION>() );
    this->m_view.setOption<VFRendering::View::CENTER_POSITION>( options().get<VFRendering::View::CENTER_POSITION>() );
    this->m_view.setOption<VFRendering::View::SYSTEM_CENTER>( options().get<VFRendering::View::SYSTEM_CENTER>() );
    this->m_view.setOption<VFRendering::View::UP_VECTOR>( options().get<VFRendering::View::UP_VECTOR>() );
    this->m_view.setOption<VFRendering::View::LIGHT_POSITION>( options().get<VFRendering::View::LIGHT_POSITION>() );
    this->m_view.setOption<VFRendering::View::VERTICAL_FIELD_OF_VIEW>(
        options().get<VFRendering::View::VERTICAL_FIELD_OF_VIEW>() );

    // TODO: this should not be necessary...
    this->updateVectorFieldGeometry();

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Colors ---
SpinWidget::Colormap SpinWidget::colormap_general() const
{
    return m_colormap_general;
}

SpinWidget::Colormap SpinWidget::colormap_arrows() const
{
    return m_colormap_arrows;
}

void SpinWidget::setCellFilter(
    int cell_a_min, int cell_a_max, int cell_b_min, int cell_b_max, int cell_c_min, int cell_c_max )
{
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    m_cell_a_min = std::max( 0, std::min( n_cells[0] - 1, cell_a_min ) );
    m_cell_a_max = std::max( 0, std::min( n_cells[0] - 1, cell_a_max ) );
    m_cell_b_min = std::max( 0, std::min( n_cells[1] - 1, cell_b_min ) );
    m_cell_b_max = std::max( 0, std::min( n_cells[1] - 1, cell_b_max ) );
    m_cell_c_min = std::max( 0, std::min( n_cells[2] - 1, cell_c_min ) );
    m_cell_c_max = std::max( 0, std::min( n_cells[2] - 1, cell_c_max ) );
}

void SpinWidget::setColormapGeneral( Colormap colormap )
{
    m_colormap_general           = colormap;
    auto colormap_implementation = getColormapRotationInverted(
        m_colormap_general, m_colormap_rotation, m_colormap_invert_z, m_colormap_invert_xy, m_colormap_cardinal_a,
        m_colormap_cardinal_b, m_colormap_cardinal_c );

    // Set overall colormap
    makeCurrent();
    this->m_view.setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation );

    // Re-set arrows map (to not overwrite it)
    this->setColormapArrows( this->colormap_arrows() );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::setColormapArrows( Colormap colormap )
{
    m_colormap_arrows            = colormap;
    auto colormap_implementation = getColormapRotationInverted(
        m_colormap_arrows, m_colormap_rotation, m_colormap_invert_z, m_colormap_invert_xy, m_colormap_cardinal_a,
        m_colormap_cardinal_b, m_colormap_cardinal_c );

    // Set arrows colormap
    makeCurrent();
    if( this->m_renderer_arrows )
        this->m_renderer_arrows->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

float SpinWidget::colormap_rotation()
{
    return this->m_colormap_rotation;
}

std::array<bool, 2> SpinWidget::colormap_inverted()
{
    return std::array<bool, 2>{ this->m_colormap_invert_z, this->m_colormap_invert_xy };
}

glm::vec3 SpinWidget::colormap_cardinal_a()
{
    return this->m_colormap_cardinal_a;
}

glm::vec3 SpinWidget::colormap_cardinal_b()
{
    return this->m_colormap_cardinal_b;
}

glm::vec3 SpinWidget::colormap_cardinal_c()
{
    return this->m_colormap_cardinal_c;
}

void SpinWidget::setColormapRotationInverted(
    int phi, bool invert_z, bool invert_xy, glm::vec3 cardinal_a, glm::vec3 cardinal_b, glm::vec3 cardinal_c )
{
    this->m_colormap_rotation   = phi;
    this->m_colormap_invert_z   = invert_z;
    this->m_colormap_invert_xy  = invert_xy;
    this->m_colormap_cardinal_a = cardinal_a;
    this->m_colormap_cardinal_b = cardinal_b;
    this->m_colormap_cardinal_c = cardinal_c;

    this->setColormapGeneral( this->colormap_general() );
    this->setColormapArrows( this->colormap_arrows() );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

std::string SpinWidget::getColormapRotationInverted(
    Colormap colormap, int phi, bool invert_z, bool invert_xy, glm::vec3 cardinal_a, glm::vec3 cardinal_b,
    glm::vec3 cardinal_c )
{
    std::locale::global(
        std::locale::classic() ); // Somewhere QT changes the locale even after initialization. As a bandaid fix we set
                                  // it here directly before we do the string conversions. This way we can at least be
                                  // sure that the decimal separator is a point and not a comma.

    int sign_z  = 1 - 2 * (int)invert_z;
    int sign_xy = 1 - 2 * (int)invert_xy;

    float P = glm::radians( (float)phi ) / 3.14159;

    std::string colormap_implementation;
    switch( colormap )
    {
        case Colormap::WHITE:
            colormap_implementation
                = VFRendering::Utilities::getColormapImplementation( VFRendering::Utilities::Colormap::WHITE );
            break;
        case Colormap::GRAY:
            colormap_implementation = R"(
        vec3 colormap(vec3 direction) {
            return vec3(0.5, 0.5, 0.5);
        }
        )";
            break;
        case Colormap::BLACK:
            colormap_implementation
                = VFRendering::Utilities::getColormapImplementation( VFRendering::Utilities::Colormap::BLACK );
            break;
        // Custom color maps not included in VFRendering:
        case Colormap::HSV:
            colormap_implementation =
                R"(
        float atan2(float y, float x) {
            return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
        }
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float hue = atan2()"
                + std::to_string( sign_xy ) + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P )
                +
                R"(/2.0;
            float saturation = projection.z * )"
                + std::to_string( sign_z ) + R"(;
            if (saturation > 0.0) {
                return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
            } else {
                return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
            }
        }
        )";
            break;
        case Colormap::HSV_NO_Z:
            colormap_implementation =
                R"(
        float atan2(float y, float x) {
            return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
        }
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float hue = atan2()"
                + std::to_string( sign_xy ) + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P )
                +
                R"(;
            return hsv2rgb(vec3(hue, 1.0, 1.0));
        }
        )";
            break;
        case Colormap::BLUE_RED:
            colormap_implementation =
                R"(
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float z_sign = projection.z * )"
                + std::to_string( sign_z ) + R"(;
            vec3 color_down = vec3(0.0, 0.0, 1.0);
            vec3 color_up = vec3(1.0, 0.0, 0.0);
            return mix(color_down, color_up, z_sign*0.5+0.5);
        }
        )";
            break;
        case Colormap::BLUE_GREEN_RED:
            colormap_implementation =
                R"(
        float atan2(float y, float x) {
            return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
        }
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float hue = 1.0/3.0-normalize(projection).z/3.0* )"
                + std::to_string( sign_z ) + R"(;
            return hsv2rgb(vec3(hue, 1.0, 1.0));
        }
        )";
            break;
        case Colormap::BLUE_WHITE_RED:
            colormap_implementation =
                R"(
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float z_sign = projection.z * )"
                + std::to_string( sign_z ) + R"(;
            if (z_sign < 0) {
                vec3 color_down = vec3(0.0, 0.0, 1.0);
                vec3 color_up = vec3(1.0, 1.0, 1.0);
                return mix(color_down, color_up, z_sign+1);
            } else {
                vec3 color_down = vec3(1.0, 1.0, 1.0);
                vec3 color_up = vec3(1.0, 0.0, 0.0);
                return mix(color_down, color_up, z_sign);
            }
        }
        )";
            break;
        // Default is regular HSV
        default:
            colormap_implementation =
                R"(
        float atan2(float y, float x) {
            return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
        }
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        vec3 colormap(vec3 direction) {
            vec3 cardinal_a = vec3()"
                + std::to_string( cardinal_a.x ) + ", " + std::to_string( cardinal_a.y ) + ", "
                + std::to_string( cardinal_a.z ) + R"();
            vec3 cardinal_b = vec3()"
                + std::to_string( cardinal_b.x ) + ", " + std::to_string( cardinal_b.y ) + ", "
                + std::to_string( cardinal_b.z ) + R"();
            vec3 cardinal_c = vec3()"
                + std::to_string( cardinal_c.x ) + ", " + std::to_string( cardinal_c.y ) + ", "
                + std::to_string( cardinal_c.z ) + R"();
            vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
            float hue = atan2()"
                + std::to_string( sign_xy ) + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P )
                +
                R"(/2.0;
            float saturation = projection.z * )"
                + std::to_string( sign_z ) + R"(;
            if (saturation > 0.0) {
                return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
            } else {
                return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
            }
        }
        )";
            break;
    }
    return colormap_implementation;
}

SpinWidget::Color SpinWidget::backgroundColor() const
{
    glm::vec3 color = options().get<VFRendering::View::Option::BACKGROUND_COLOR>();
    if( color == glm::vec3{ 0, 0, 0 } )
        return Color::BLACK;
    else if( color == glm::vec3{ 0.5, 0.5, 0.5 } )
        return Color::GRAY;
    else if( color == glm::vec3{ 1, 1, 1 } )
        return Color::WHITE;
    else
        return Color::OTHER;
}

void SpinWidget::setBackgroundColor( Color background_color )
{
    glm::vec3 color;
    if( background_color == Color::BLACK )
        color = { 0, 0, 0 };
    else if( background_color == Color::GRAY )
        color = { 0.5, 0.5, 0.5 };
    else if( background_color == Color::WHITE )
        color = { 1, 1, 1 };
    makeCurrent();
    this->m_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>( color );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

SpinWidget::Color SpinWidget::boundingBoxColor() const
{
    glm::vec3 color = options().get<VFRendering::BoundingBoxRenderer::Option::COLOR>();
    if( color == glm::vec3{ 0, 0, 0 } )
        return Color::BLACK;
    else if( color == glm::vec3{ 0.5, 0.5, 0.5 } )
        return Color::GRAY;
    else if( color == glm::vec3{ 1, 1, 1 } )
        return Color::WHITE;
    else
        return Color::OTHER;
}

void SpinWidget::setBoundingBoxColor( Color bounding_box_color )
{
    glm::vec3 color;
    if( bounding_box_color == Color::BLACK )
        color = { 0, 0, 0 };
    else if( bounding_box_color == Color::GRAY )
        color = { 0.5, 0.5, 0.5 };
    else if( bounding_box_color == Color::WHITE )
        color = { 1, 1, 1 };
    makeCurrent();
    this->m_view.setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>( color );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::updateBoundingBoxIndicators()
{
    bool periodical[3];
    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    glm::vec3 bounding_box_center = { ( bounds_min[0] + bounds_max[0] ) / 2, ( bounds_min[1] + bounds_max[1] ) / 2,
                                      ( bounds_min[2] + bounds_max[2] ) / 2 };
    glm::vec3 bounding_box_side_lengths
        = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

    float indi_length            = glm::length( bounds_max - bounds_min ) * 0.05;
    int indi_dashes              = 5;
    float indi_dashes_per_length = (float)indi_dashes / indi_length;

    Hamiltonian_Get_Boundary_Conditions( this->state.get(), periodical );
    glm::vec3 indis{ indi_length * periodical[0], indi_length * periodical[1], indi_length * periodical[2] };

    this->m_renderer_boundingbox
        = std::make_shared<VFRendering::BoundingBoxRenderer>( VFRendering::BoundingBoxRenderer::forCuboid(
            m_view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length ) );

    this->enableSystem( this->show_arrows, this->show_boundingbox, this->show_surface, this->show_isosurface );
}

//////////////////////////////////////////////////////////////////////////////////////
///// --- Camera ---
void SpinWidget::cycleCamera()
{
    if( this->m_camera_projection_perspective )
    {
        this->m_camera_projection_perspective = false;
    }
    else
    {
        this->m_camera_projection_perspective = true;
    }
    this->setVerticalFieldOfView( this->user_fov );
}

void SpinWidget::setCameraToDefault()
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        float camera_distance = 30.0f;
        auto center_position  = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        auto camera_position  = center_position + camera_distance * glm::vec3( 0, 0, 1 );
        auto up_vector        = glm::vec3( 0, 1, 0 );

        VFRendering::Options options;
        options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
        options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
        options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
        this->m_view.updateOptions( options );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::setCameraToX( bool inverted )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        float camera_distance = glm::length(
            options().get<VFRendering::View::Option::CENTER_POSITION>()
            - options().get<VFRendering::View::Option::CAMERA_POSITION>() );
        auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        auto camera_position = center_position;
        auto up_vector       = glm::vec3( 0, 0, 1 );

        if( !inverted )
        {
            camera_position += camera_distance * glm::vec3( 1, 0, 0 );
        }
        else
        {
            camera_position -= camera_distance * glm::vec3( 1, 0, 0 );
        }

        VFRendering::Options options;
        options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
        options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
        options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
        this->m_view.updateOptions( options );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::setCameraToY( bool inverted )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        float camera_distance = glm::length(
            options().get<VFRendering::View::Option::CENTER_POSITION>()
            - options().get<VFRendering::View::Option::CAMERA_POSITION>() );
        auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        auto camera_position = center_position;
        auto up_vector       = glm::vec3( 0, 0, 1 );

        if( !inverted )
            camera_position += camera_distance * glm::vec3( 0, -1, 0 );
        else
            camera_position -= camera_distance * glm::vec3( 0, -1, 0 );

        VFRendering::Options options;
        options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
        options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
        options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
        this->m_view.updateOptions( options );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::setCameraToZ( bool inverted )
{
    float camera_distance = glm::length(
        options().get<VFRendering::View::Option::CENTER_POSITION>()
        - options().get<VFRendering::View::Option::CAMERA_POSITION>() );
    auto center_position = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = center_position;
    auto up_vector       = glm::vec3( 0, 1, 0 );

    if( !inverted )
        camera_position += camera_distance * glm::vec3( 0, 0, 1 );
    else
        camera_position -= camera_distance * glm::vec3( 0, 0, 1 );

    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
    options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
    options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
    this->m_view.updateOptions( options );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

void SpinWidget::setCameraPosition( const glm::vec3 & camera_position )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        this->m_view.setOption<VFRendering::View::Option::CAMERA_POSITION>( system_center + camera_position );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::setCameraFocus( const glm::vec3 & center_position )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        this->m_view.setOption<VFRendering::View::Option::CENTER_POSITION>( system_center + center_position );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

void SpinWidget::setCameraUpVector( const glm::vec3 & up_vector )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        this->m_view.setOption<VFRendering::View::Option::UP_VECTOR>( up_vector );

        QTimer::singleShot( 1, this, SLOT( update() ) );
    }
}

glm::vec3 SpinWidget::getCameraPositon()
{
    auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    return options().get<VFRendering::View::Option::CAMERA_POSITION>() - system_center;
}

glm::vec3 SpinWidget::getCameraFocus()
{
    auto system_center = options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    return options().get<VFRendering::View::Option::CENTER_POSITION>() - system_center;
}

glm::vec3 SpinWidget::getCameraUpVector()
{
    return options().get<VFRendering::View::Option::UP_VECTOR>();
}

float SpinWidget::verticalFieldOfView() const
{
    return this->user_fov;
}

void SpinWidget::setVerticalFieldOfView( float vertical_field_of_view )
{
    this->user_fov = vertical_field_of_view;
    if( !this->m_camera_projection_perspective )
    {
        vertical_field_of_view = 0;
    }

    // Calculate new camera position
    float scale = 1;
    float fov   = options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
    if( fov > 0 && vertical_field_of_view > 0 )
    {
        scale = std::tan( glm::radians( fov ) / 2.0 ) / std::tan( glm::radians( vertical_field_of_view ) / 2.0 );
        setCameraPosition( getCameraPositon() * scale );
    }
    else if( fov > 0 )
    {
        scale = std::tan( glm::radians( fov ) / 2.0 );
        setCameraPosition( getCameraPositon() * scale );
    }
    else if( vertical_field_of_view > 0 )
    {
        scale = 1.0 / std::tan( glm::radians( vertical_field_of_view ) / 2.0 );
        setCameraPosition( getCameraPositon() * scale );
    }

    // Set new FOV
    makeCurrent();
    this->m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( vertical_field_of_view );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

bool SpinWidget::cameraProjection()
{
    return this->m_camera_projection_perspective;
}

void SpinWidget::setCameraProjection( bool perspective )
{
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        this->m_camera_projection_perspective = perspective;
        this->setVerticalFieldOfView( this->user_fov );
    }
}

bool SpinWidget::getCameraRotationType()
{
    return this->m_camera_rotate_free;
}
void SpinWidget::setCameraRotationType( bool free )
{
    this->m_camera_rotate_free = free;
}

/////////////// lighting
glm::vec3 from_spherical( float theta, float phi )
{
    float x = glm::sin( glm::radians( theta ) ) * glm::cos( glm::radians( phi ) );
    float y = glm::sin( glm::radians( theta ) ) * glm::sin( glm::radians( phi ) );
    float z = glm::cos( glm::radians( theta ) );
    return glm::vec3{ x, y, z };
}
void SpinWidget::setLightPosition( float theta, float phi )
{
    this->m_light_theta = theta;
    this->m_light_phi   = phi;
    glm::vec3 v_light   = glm::normalize( from_spherical( theta, phi ) ) * 1000.0f;
    this->m_view.setOption<VFRendering::View::Option::LIGHT_POSITION>( v_light );

    QTimer::singleShot( 1, this, SLOT( update() ) );
}

std::array<float, 2> SpinWidget::getLightPosition()
{
    return std::array<float, 2>{ m_light_theta, m_light_phi };
}

// -----------------------------------------------------------------------------------
// --------------------- Persistent Settings -----------------------------------------
// -----------------------------------------------------------------------------------

void SpinWidget::writeSettings()
{
    QSettings settings( "Spirit Code", "Spirit" );

    settings.beginGroup( "General" );
    // VisMode
    settings.setValue( "Mode", (int)( this->visualizationMode() ) );
    // Sphere Point Size
    settings.setValue( "SpherePointSize1", (int)( this->spherePointSizeRange().x * 100 ) );
    settings.setValue( "SpherePointSize2", (int)( this->spherePointSizeRange().y * 100 ) );
    // System
    settings.setValue( "Cycle Index", this->idx_cycle );
    settings.setValue( "Show Arrows", this->user_show_arrows );
    settings.setValue( "Show Bounding Box", this->user_show_boundingbox );
    settings.setValue( "Show Surface", this->user_show_surface );
    settings.setValue( "Show Isosurface", this->user_show_isosurface );
    // MiniView
    settings.setValue( "Show MiniView", this->show_miniview );
    settings.setValue( "MiniView Position", (int)this->m_location_miniview );
    // Coordinate System
    settings.setValue( "Show Coordinate System", this->show_coordinatesystem );
    settings.setValue( "Coordinate System Position", (int)this->m_location_coordinatesystem );
    settings.endGroup();

    // Arrows
    settings.beginGroup( "Arrows" );
    settings.setValue( "Size", (int)( this->arrowSize() * 100 ) );
    settings.setValue( "LOD", this->arrowLOD() );
    settings.endGroup();

    // Colors
    settings.beginGroup( "Colors" );
    settings.setValue( "Background Color", (int)backgroundColor() );
    settings.setValue( "Colormap General", (int)colormap_general() );
    settings.setValue( "Colormap Arrows", (int)colormap_arrows() );
    settings.setValue( "Colormap_invert_z", m_colormap_invert_z );
    settings.setValue( "Colormap_invert_xy", m_colormap_invert_xy );
    settings.setValue( "Colormap_rotation", m_colormap_rotation );
    settings.setValue( "Colormap_cardinal_a_x", m_colormap_cardinal_a.x );
    settings.setValue( "Colormap_cardinal_a_y", m_colormap_cardinal_a.y );
    settings.setValue( "Colormap_cardinal_a_z", m_colormap_cardinal_a.z );
    settings.setValue( "Colormap_cardinal_b_x", m_colormap_cardinal_b.x );
    settings.setValue( "Colormap_cardinal_b_y", m_colormap_cardinal_b.y );
    settings.setValue( "Colormap_cardinal_b_z", m_colormap_cardinal_b.z );
    settings.setValue( "Colormap_cardinal_c_x", m_colormap_cardinal_c.x );
    settings.setValue( "Colormap_cardinal_c_y", m_colormap_cardinal_c.y );
    settings.setValue( "Colormap_cardinal_c_z", m_colormap_cardinal_c.z );
    settings.endGroup();

    // Camera
    settings.beginGroup( "Camera" );
    glm::vec3 camera_position;
    glm::vec3 center_position;
    glm::vec3 up_vector;
    bool perspective;
    if( this->m_interactionmode == InteractionMode::REGULAR )
    {
        perspective     = this->cameraProjection();
        camera_position = this->getCameraPositon();
        center_position = this->getCameraFocus();
        up_vector       = this->getCameraUpVector();
    }
    else
    {
        perspective     = this->regular_mode_perspective;
        camera_position = this->regular_mode_cam_pos;
        center_position = this->regular_mode_cam_focus;
        up_vector       = this->regular_mode_cam_up;
    }
    settings.beginWriteArray( "position" );
    for( int dim = 0; dim < 3; ++dim )
    {
        settings.setArrayIndex( dim );
        settings.setValue( "vecp", (int)( 100 * camera_position[dim] ) );
    }
    settings.endArray();
    settings.beginWriteArray( "center" );
    for( int dim = 0; dim < 3; ++dim )
    {
        settings.setArrayIndex( dim );
        settings.setValue( "vecc", (int)( 100 * center_position[dim] ) );
    }
    settings.endArray();
    settings.beginWriteArray( "up" );
    for( int dim = 0; dim < 3; ++dim )
    {
        settings.setArrayIndex( dim );
        settings.setValue( "vecu", (int)( 100 * up_vector[dim] ) );
    }
    settings.endArray();
    settings.setValue( "FOV", (int)( this->user_fov * 100 ) );
    settings.setValue( "perspective projection", perspective );
    settings.setValue( "free rotation", this->m_camera_rotate_free );
    settings.endGroup();

    // Light
    settings.beginGroup( "Light" );
    settings.setValue( "theta", (int)m_light_theta );
    settings.setValue( "phi", (int)m_light_phi );
    settings.endGroup();
}

void SpinWidget::readSettings()
{
    makeCurrent();
    QSettings settings( "Spirit Code", "Spirit" );

    settings.beginGroup( "General" );
    // VisMode
    this->visMode = VisualizationMode( settings.value( "Mode", 0 ).toInt() );
    // Sphere Point Size
    this->setSpherePointSizeRange( { ( settings.value( "SpherePointSize1", 100 ).toInt() / 100.0f ),
                                     ( settings.value( "SpherePointSize2", 100 ).toInt() / 100.0f ) } );
    // System
    this->idx_cycle             = 0; // settings.value("Cycle Index").toInt();
    this->user_show_arrows      = settings.value( "Show Arrows", true ).toBool();
    this->user_show_boundingbox = settings.value( "Show Bounding Box", true ).toBool();
    this->user_show_surface     = settings.value( "Show Surface", false ).toBool();
    this->user_show_isosurface  = settings.value( "Show Isosurface", false ).toBool();
    // MiniView
    this->show_miniview = settings.value( "Show MiniView", true ).toBool();
    this->m_location_miniview
        = WidgetLocation( settings.value( "MiniView Position", (int)WidgetLocation::BOTTOM_LEFT ).toInt() );
    // Coordinate System
    this->show_coordinatesystem = settings.value( "Show Coordinate System", true ).toBool();
    this->m_location_coordinatesystem
        = WidgetLocation( settings.value( "Coordinate System Position", (int)WidgetLocation::BOTTOM_RIGHT ).toInt() );
    settings.endGroup();

    // Arrows
    if( settings.childGroups().contains( "Arrows" ) )
    {
        settings.beginGroup( "Arrows" );
        // Projection
        this->setArrows( (float)( settings.value( "Size" ).toInt() / 100.0f ), settings.value( "LOD" ).toInt() );
        settings.endGroup();
    }

    // Colors
    if( settings.childGroups().contains( "Colors" ) )
    {
        settings.beginGroup( "Colors" );
        int background_color = settings.value( "Background Color" ).toInt();
        this->setBackgroundColor( (Color)background_color );
        if( background_color == 2 )
            this->setBoundingBoxColor( (Color)0 );
        else
            this->setBoundingBoxColor( (Color)2 );
        int map_arrows = settings.value( "Colormap Arrows" ).toInt();
        this->setColormapArrows( (Colormap)map_arrows );
        int map_genreal = settings.value( "Colormap General" ).toInt();
        this->setColormapGeneral( (Colormap)map_genreal );
        bool invert_z  = settings.value( "Colormap_invert_z" ).toInt();
        bool invert_xy = settings.value( "Colormap_invert_xy" ).toInt();
        int phi        = settings.value( "Colormap_rotation" ).toInt();
        glm::vec3 cardinal_a{ 1, 0, 0 }, cardinal_b{ 0, 1, 0 }, cardinal_c{ 0, 0, 1 };
        cardinal_a.x = settings.value( "Colormap_cardinal_a_x" ).toFloat();
        cardinal_a.y = settings.value( "Colormap_cardinal_a_y" ).toFloat();
        cardinal_a.z = settings.value( "Colormap_cardinal_a_z" ).toFloat();
        cardinal_b.x = settings.value( "Colormap_cardinal_b_x" ).toFloat();
        cardinal_b.y = settings.value( "Colormap_cardinal_b_y" ).toFloat();
        cardinal_b.z = settings.value( "Colormap_cardinal_b_z" ).toFloat();
        cardinal_c.x = settings.value( "Colormap_cardinal_c_x" ).toFloat();
        cardinal_c.y = settings.value( "Colormap_cardinal_c_y" ).toFloat();
        cardinal_c.z = settings.value( "Colormap_cardinal_c_z" ).toFloat();
        this->setColormapRotationInverted( phi, invert_z, invert_xy, cardinal_a, cardinal_b, cardinal_c );
        settings.endGroup();
    }
    else
    {
        glm::vec3 cardinal_a{ 1, 0, 0 }, cardinal_b{ 0, 1, 0 }, cardinal_c{ 0, 0, 1 };
        this->setColormapRotationInverted( 0, false, false, cardinal_a, cardinal_b, cardinal_c );
    }

    // Camera
    settings.beginGroup( "Camera" );
    this->user_fov                        = (float)( settings.value( "FOV", 45 * 100 ).toInt() / 100.0f );
    this->m_camera_projection_perspective = settings.value( "perspective projection", true ).toBool();
    this->regular_mode_perspective        = this->m_camera_projection_perspective;
    this->m_camera_rotate_free            = settings.value( "free rotation", 0 ).toBool();
    this->m_view.setOption<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>(
        this->m_camera_projection_perspective * this->user_fov );
    glm::vec3 camera_position, center_position, up_vector;
    settings.endGroup();
    this->setCameraToDefault();
    if( settings.childGroups().contains( "Camera" ) )
    {
        settings.beginGroup( "Camera" );
        settings.beginReadArray( "position" );
        for( int dim = 0; dim < 3; ++dim )
        {
            settings.setArrayIndex( dim );
            camera_position[dim] = (float)( settings.value( "vecp" ).toInt() / 100.0f );
        }
        settings.endArray();
        this->setCameraPosition( camera_position );
        settings.beginReadArray( "center" );
        for( int dim = 0; dim < 3; ++dim )
        {
            settings.setArrayIndex( dim );
            center_position[dim] = (float)( settings.value( "vecc" ).toInt() / 100.0f );
        }
        settings.endArray();
        this->setCameraFocus( center_position );

        settings.beginReadArray( "up" );
        for( int dim = 0; dim < 3; ++dim )
        {
            settings.setArrayIndex( dim );
            up_vector[dim] = (float)( settings.value( "vecu" ).toInt() / 100.0f );
        }
        settings.endArray();
        this->setCameraUpVector( up_vector );
        settings.endGroup();
    }

    // Light
    if( settings.childGroups().contains( "Light" ) )
    {
        settings.beginGroup( "Light" );
        this->m_light_theta = settings.value( "theta" ).toInt();
        this->m_light_phi   = settings.value( "phi" ).toInt();
        this->setLightPosition( m_light_theta, m_light_phi );
        settings.endGroup();
    }
}

void SpinWidget::closeEvent( QCloseEvent * event )
{
    writeSettings();
    event->accept();
}
