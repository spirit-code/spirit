#include <glad/glad.h>

#include <rendering_layer.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>

#include <nlohmann/json.hpp>

#include <fmt/format.h>

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace ui
{

RenderingLayer::RenderingLayer( ui::UiSharedState & ui_shared_state, std::shared_ptr<State> state )
        : ui_shared_state( ui_shared_state ), state( state )
{
    this->reset_camera();
}

void RenderingLayer::draw( int display_w, int display_h )
{
    if( !gl_initialized_ )
        initialize_gl();

    if( needs_data_ )
    {
        update_vf_directions();
        needs_data_   = false;
        needs_redraw_ = true;
    }

    if( Simulation_Running_On_Image( state.get() ) )
        needs_redraw_ = true;

    bool update_widgets = false;

    if( coordinatesystem_renderer_widget->show_ != show_coordinatesystem_ )
    {
        show_coordinatesystem_ = coordinatesystem_renderer_widget->show_;
        update_widgets         = true;
    }
    if( boundingbox_renderer_widget->show_ != show_boundingbox_ )
    {
        show_boundingbox_ = boundingbox_renderer_widget->show_;
        update_widgets    = true;
    }

    for( auto & renderer_widget : renderer_widgets_shown )
    {
        if( !renderer_widget->show_ )
            update_widgets = true;
    }
    for( auto & renderer_widget : renderer_widgets_not_shown )
    {
        if( renderer_widget->show_ )
            update_widgets = true;
    }

    for( auto renderer_widget_iterator = renderer_widgets.begin(); renderer_widget_iterator != renderer_widgets.end(); )
    {
        if( ( *renderer_widget_iterator )->remove_ )
        {
            renderer_widget_iterator = renderer_widgets.erase( renderer_widget_iterator );
            update_widgets           = true;
        }
        else
            ++renderer_widget_iterator;
    }

    if( update_widgets )
    {
        needs_redraw_ = true;
        renderer_widgets_shown.resize( 0 );
        renderer_widgets_not_shown.resize( 0 );

        for( auto & renderer_widget : renderer_widgets )
        {
            if( renderer_widget->show_ )
                renderer_widgets_shown.push_back( renderer_widget );
            else
                renderer_widgets_not_shown.push_back( renderer_widget );
        }

        update_renderers();
    }

    if( needs_redraw_ )
    {
        view.setFramebufferSize( float( display_w ), float( display_h ) );
        needs_redraw_ = false;
    }

    for( auto & vfr_update_call : this->vfr_update_deque )
        vfr_update_call();
    this->vfr_update_deque.resize( 0 );

    view.draw();
}

void RenderingLayer::screenshot_png( std::string filename )
{
    auto display_w = view.getFramebufferSize().x;
    auto display_h = view.getFramebufferSize().y;

    std::vector<GLubyte> pixels = std::vector<GLubyte>( 3 * display_w * display_h, 0 );
    glPixelStorei( GL_PACK_ALIGNMENT, 1 );
    glReadPixels( 0, 0, display_w, display_h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data() );

    // The order of pixel rows in OpenGL and STB is opposite, so we invert the ordering
    for( int line = 0; line < display_h / 2; ++line )
    {
        std::swap_ranges(
            pixels.begin() + 3 * display_w * line, pixels.begin() + 3 * display_w * ( line + 1 ),
            pixels.begin() + 3 * display_w * ( display_h - line - 1 ) );
    }

    stbi_write_png( fmt::format( "{}.png", filename ).c_str(), display_w, display_h, 3, pixels.data(), display_w * 3 );
}

void RenderingLayer::needs_redraw()
{
    needs_redraw_ = true;
}

void RenderingLayer::needs_data()
{
    needs_data_ = true;
}

void RenderingLayer::initialize_gl()
{
    if( ui_shared_state.dark_mode )
        set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>( glm::vec3{
            ui_shared_state.background_dark[0],
            ui_shared_state.background_dark[1],
            ui_shared_state.background_dark[2],
        } );
    else
        set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>( glm::vec3{
            ui_shared_state.background_light[0],
            ui_shared_state.background_light[1],
            ui_shared_state.background_light[2],
        } );

    set_view_option<VFRendering::View::Option::LIGHT_POSITION>( glm::vec3{
        -1000 * ui_shared_state.light_direction[0],
        -1000 * ui_shared_state.light_direction[1],
        -1000 * ui_shared_state.light_direction[2],
    } );

    if( !boundingbox_renderer_widget )
        boundingbox_renderer_widget = std::make_shared<BoundingBoxRendererWidget>(
            state, view, vectorfield, ui_shared_state, vfr_update_deque );
    if( !coordinatesystem_renderer_widget )
        coordinatesystem_renderer_widget
            = std::make_shared<CoordinateSystemRendererWidget>( state, view, vfr_update_deque );

    this->update_vf_geometry();
    this->update_vf_directions();

    // Defaults
    if( renderer_widgets.empty() )
    {
        renderer_widgets.push_back(
            std::make_shared<ArrowRendererWidget>( state, view, vectorfield, vfr_update_deque ) );
        renderer_widgets.push_back(
            std::make_shared<IsosurfaceRendererWidget>( state, view, vectorfield, vfr_update_deque ) );
        renderer_widgets[0]->id   = 0;
        renderer_widgets[1]->id   = 1;
        this->renderer_id_counter = 2;
    }

    for( auto & renderer_widget : renderer_widgets )
    {
        if( renderer_widget->show_ )
            renderer_widgets_shown.push_back( renderer_widget );
        else
            renderer_widgets_not_shown.push_back( renderer_widget );
    }

    this->update_renderers();

    boundingbox_renderer_widget->apply_settings();
    coordinatesystem_renderer_widget->apply_settings();
    for( auto & renderer_widget : renderer_widgets )
        renderer_widget->apply_settings();

    gl_initialized_ = true;
}

void RenderingLayer::update_theme()
{
    if( ui_shared_state.dark_mode )
    {
        set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>( glm::vec3{
            ui_shared_state.background_dark[0],
            ui_shared_state.background_dark[1],
            ui_shared_state.background_dark[2],
        } );
    }
    else
    {
        set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>( glm::vec3{
            ui_shared_state.background_light[0],
            ui_shared_state.background_light[1],
            ui_shared_state.background_light[2],
        } );
    }
    boundingbox_renderer_widget->apply_settings();
}

void RenderingLayer::update_renderers()
{
    std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> total_renderers;

    std::vector<std::shared_ptr<VFRendering::RendererBase>> system_renderers;
    if( boundingbox_renderer_widget->show_ )
        system_renderers.push_back( boundingbox_renderer_widget->renderer );
    for( auto & renderer_widget : renderer_widgets )
    {
        if( renderer_widget->show_ )
            system_renderers.push_back( renderer_widget->renderer );
    }

    combined_renderer = std::make_shared<VFRendering::CombinedRenderer>( view, system_renderers );

    update_renderers_from_layout();
}

void RenderingLayer::update_renderers_from_layout()
{
    std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> total_renderers;

    total_renderers.push_back( { combined_renderer, rendering_layout } );

    if( coordinatesystem_renderer_widget->show_ )
    {
        float pos_x  = rendering_layout[0] + rendering_layout[2] - 0.2f;
        float size_x = 0.2f;
        float pos_y  = 0;
        float size_y = 0.2f;
        total_renderers.push_back( { coordinatesystem_renderer_widget->renderer, { pos_x, pos_y, size_x, size_y } } );
    }

    view.renderers( total_renderers );
}

void RenderingLayer::update_vf_geometry()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

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
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis
                              + n_cell_atoms * n_cell_step
                                    * ( +cell_a + n_cells[0] * cell_b + n_cells[0] * n_cells[1] * cell_c );
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
        if( n_cell_step > 1
            && ( n_cells[0] / n_cell_step < 2 || n_cells[1] / n_cell_step < 2 || n_cells[2] / n_cell_step < 2 ) )
        {
            geometry = VFRendering::Geometry( positions, {}, {}, true );
        }
        else
        {
            const std::array<VFRendering::Geometry::index_type, 4> * tetrahedra_indices_ptr = nullptr;

            int num_tetrahedra = Geometry_Get_Tetrahedra(
                state.get(), reinterpret_cast<const int **>( &tetrahedra_indices_ptr ), n_cell_step );
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
        for( int i = 1, j = 0; i < nos && j < 2; ++i )
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
        glm::vec3 normal = glm::normalize( glm::cross( basis[0], basis[1] ) );
        // By default, +z is up, which is where we want the normal oriented towards
        if( glm::dot( normal, glm::vec3{ 0, 0, 1 } ) < 1e-6 )
            normal = -normal;

        // Rectilinear with one basis atom
        if( n_cell_atoms == 1 && std::abs( glm::dot( basis[0], basis[1] ) ) < 1e-6 )
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
            int num_triangles                                                             = Geometry_Get_Triangulation(
                state.get(), reinterpret_cast<const int **>( &triangle_indices_ptr ), n_cell_step );
            std::vector<std::array<VFRendering::Geometry::index_type, 3>> triangle_indices(
                triangle_indices_ptr, triangle_indices_ptr + num_triangles );
            geometry = VFRendering::Geometry( positions, triangle_indices, {}, true );
            for( int i = 0; i < nos_draw; ++i )
                positions[i] = positions[i] - normal;
            geometry_surf2D = VFRendering::Geometry( positions, triangle_indices, {}, true );
        }

        // Update the vectorfield geometry
        vectorfield_surf2D.updateGeometry( geometry_surf2D );
    }
    else
    {
        geometry = VFRendering::Geometry( positions, {}, {}, true );
    }

    // Update the vectorfield
    vectorfield.updateGeometry( geometry );
    needs_data_ = true;

    update_visibility();
    update_boundingbox();
    update_renderers();
}

void RenderingLayer::update_visibility()
{
    const float epsilon = 1e-5;

    float b_min[3], b_max[3], b_range[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );

    float filter_pos_min[3], filter_pos_max[3];
    float filter_dir_min[3], filter_dir_max[3];
    for( int dim = 0; dim < 3; ++dim )
    {
        b_range[dim]        = b_max[dim] - b_min[dim];
        filter_pos_min[dim] = b_min[dim] + this->filter_position_min[dim] * b_range[dim] - epsilon;
        filter_pos_max[dim] = b_max[dim] + ( this->filter_position_max[dim] - 1 ) * b_range[dim] + epsilon;

        filter_dir_min[dim] = this->filter_direction_min[dim] - epsilon;
        filter_dir_max[dim] = this->filter_direction_max[dim] + epsilon;
    }
    std::string is_visible = fmt::format(
        R"(
            bool is_visible(vec3 position, vec3 direction)
            {{
                float x_min_pos = {:.6f};
                float x_max_pos = {:.6f};
                bool is_visible_x_pos = position.x <= x_max_pos && position.x >= x_min_pos;

                float y_min_pos = {:.6f};
                float y_max_pos = {:.6f};
                bool is_visible_y_pos = position.y <= y_max_pos && position.y >= y_min_pos;

                float z_min_pos = {:.6f};
                float z_max_pos = {:.6f};
                bool is_visible_z_pos = position.z <= z_max_pos && position.z >= z_min_pos;

                float x_min_dir = {:.6f};
                float x_max_dir = {:.6f};
                bool is_visible_x_dir = direction.x <= x_max_dir && direction.x >= x_min_dir;

                float y_min_dir = {:.6f};
                float y_max_dir = {:.6f};
                bool is_visible_y_dir = direction.y <= y_max_dir && direction.y >= y_min_dir;

                float z_min_dir = {:.6f};
                float z_max_dir = {:.6f};
                bool is_visible_z_dir = direction.z <= z_max_dir && direction.z >= z_min_dir;

                return is_visible_x_pos && is_visible_y_pos && is_visible_z_pos && is_visible_x_dir && is_visible_y_dir && is_visible_z_dir;
            }}
            )",
        filter_pos_min[0], filter_pos_max[0], filter_pos_min[1], filter_pos_max[1], filter_pos_min[2],
        filter_pos_max[2], this->filter_direction_min[0], this->filter_direction_max[0], this->filter_direction_min[1],
        this->filter_direction_max[1], this->filter_direction_min[2], this->filter_direction_max[2] );

    this->set_view_option<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible );
}

void RenderingLayer::update_boundingbox()
{
    boundingbox_renderer_widget->update_geometry();
    update_renderers();
}

void RenderingLayer::update_vf_directions()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Directions of the vectorfield
    std::vector<glm::vec3> directions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Directions
    //        get pointer
    scalar * spins;
    int * atom_types;
    atom_types = Geometry_Get_Atom_Types( state.get() );
    // if( this->m_source == 0 )
    spins = System_Get_Spin_Directions( state.get() );
    // else if( this->m_source == 1 )
    //     spins = System_Get_Effective_Field( state.get() );
    // else spins = System_Get_Spin_Directions( state.get() );

    //        copy
    /*positions.assign(spin_pos, spin_pos + 3*nos);
    directions.assign(spins, spins + 3*nos);*/
    int icell = 0;
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis + n_cell_atoms * cell_a * n_cell_step
                              + n_cell_atoms * n_cells[0] * cell_b * n_cell_step
                              + n_cell_atoms * n_cells[0] * n_cells[1] * cell_c * n_cell_step;
                    // std::cerr << idx << " " << icell << std::endl;
                    directions[icell] = glm::vec3( spins[3 * idx], spins[1 + 3 * idx], spins[2 + 3 * idx] );
                    if( atom_types[idx] < 0 )
                        directions[icell] *= 0;
                    ++icell;
                }
            }
        }
    }
    // //        rescale if effective field
    // if( this->m_source == 1 )
    // {
    //     float max_length = 0;
    //     for( auto direction : directions )
    //     {
    //         max_length = std::max( max_length, glm::length( direction ) );
    //     }
    //     if( max_length > 0 )
    //     {
    //         for( auto & direction : directions )
    //         {
    //             direction /= max_length;
    //         }
    //     }
    // }

    // Update the vectorfield
    vectorfield.updateVectors( directions );

    if( Geometry_Get_Dimensionality( state.get() ) == 2 )
        vectorfield_surf2D.updateVectors( directions );
}

void RenderingLayer::reset_camera()
{
    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min      = glm::make_vec3( b_min );
    glm::vec3 bounds_max      = glm::make_vec3( b_max );
    glm::vec3 center_position = ( bounds_min + bounds_max ) * 0.5f;
    float camera_distance     = glm::distance( bounds_min, bounds_max );
    auto camera_position      = center_position + camera_distance * glm::vec3( 0, 0, 1 );
    auto up_vector            = glm::vec3( 0, 1, 0 );

    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>( center_position );
    options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( 45 );
    options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
    options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
    options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
    this->view.updateOptions( options );

    this->needs_redraw();
}

// Uses the camera's current FOV and distance from the system center to update the camera position for
// a new FOV without changing the perceived distance from the system center.
void change_camera_position_from_fov(
    float previous_fov, float new_fov, glm::vec3 system_center, glm::vec3 & camera_position )
{
    float scale = 1;
    if( previous_fov > 0 && new_fov > 0 )
        scale = std::tan( glm::radians( previous_fov ) / 2.0 ) / std::tan( glm::radians( new_fov ) / 2.0 );
    else if( previous_fov > 0 )
        scale = std::tan( glm::radians( previous_fov ) / 2.0 );
    else if( new_fov > 0 )
        scale = 1.0 / std::tan( glm::radians( new_fov ) / 2.0 );
    auto new_camera_position = system_center + ( camera_position - system_center ) * scale;
    camera_position          = new_camera_position;
}

void RenderingLayer::set_camera_fov( float new_fov )
{
    if( ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
        return;

    float previous_fov   = view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
    auto system_center   = view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
    auto camera_position = view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
    change_camera_position_from_fov( previous_fov, new_fov, system_center, camera_position );

    // Update
    set_view_option<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
    set_view_option<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( new_fov );
}

void RenderingLayer::set_camera_orthographic( bool orthographic )
{
    if( ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
        return;

    if( ui_shared_state.camera_is_orthographic && !orthographic )
    {
        this->set_camera_fov( ui_shared_state.camera_perspective_fov );
    }
    else if( !ui_shared_state.camera_is_orthographic && orthographic )
    {
        ui_shared_state.camera_perspective_fov
            = view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
        this->set_camera_fov( 0 );
    }

    ui_shared_state.camera_is_orthographic = orthographic;
}

void RenderingLayer::set_interaction_mode( UiSharedState::InteractionMode interaction_mode )
{
    if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR
        && interaction_mode != UiSharedState::InteractionMode::REGULAR )
    {
        // Save "regular camera" data
        auto cam_pos = view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
        auto cen_pos = view.options().get<VFRendering::View::Option::CENTER_POSITION>();
        auto sys_cen = view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
        auto up_vec  = view.options().get<VFRendering::View::Option::UP_VECTOR>();
        for( int dim = 0; dim < 3; ++dim )
        {
            ui_shared_state.regular_interaction_camera_pos[dim] = cam_pos[dim];
            ui_shared_state.regular_interaction_center_pos[dim] = cen_pos[dim];
            ui_shared_state.regular_interaction_sys_center[dim] = sys_cen[dim];
            ui_shared_state.regular_interaction_camera_up[dim]  = up_vec[dim];
        }
        if( !ui_shared_state.camera_is_orthographic )
            ui_shared_state.camera_perspective_fov
                = view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();

        // The "interaction mode camera" is orthographic, so we calculate the new distance
        float prev_fov = view.options().get<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>();
        change_camera_position_from_fov( prev_fov, 0, sys_cen, cam_pos );

        float camera_distance = glm::length( cen_pos - cam_pos );
        auto camera_position  = sys_cen + camera_distance * glm::vec3( 0, 0, 1 );

        // Update
        VFRendering::Options options;
        options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
        options.set<VFRendering::View::Option::CENTER_POSITION>( sys_cen );
        options.set<VFRendering::View::Option::UP_VECTOR>( { 0, 1, 0 } );
        options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( 0 );
        view.updateOptions( options );
        needs_redraw();
    }
    else if(
        ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR
        && interaction_mode == UiSharedState::InteractionMode::REGULAR )
    {
        // Restore "regular camera" data
        VFRendering::Options options;
        options.set<VFRendering::View::Option::CENTER_POSITION>(
            { ui_shared_state.regular_interaction_center_pos[0], ui_shared_state.regular_interaction_center_pos[1],
              ui_shared_state.regular_interaction_center_pos[2] } );
        options.set<VFRendering::View::Option::SYSTEM_CENTER>( { ui_shared_state.regular_interaction_sys_center[0],
                                                                 ui_shared_state.regular_interaction_sys_center[1],
                                                                 ui_shared_state.regular_interaction_sys_center[2] } );
        options.set<VFRendering::View::Option::UP_VECTOR>( { ui_shared_state.regular_interaction_camera_up[0],
                                                             ui_shared_state.regular_interaction_camera_up[1],
                                                             ui_shared_state.regular_interaction_camera_up[2] } );
        options.set<VFRendering::View::Option::CAMERA_POSITION>(
            { ui_shared_state.regular_interaction_camera_pos[0], ui_shared_state.regular_interaction_camera_pos[1],
              ui_shared_state.regular_interaction_camera_pos[2] } );

        if( !ui_shared_state.camera_is_orthographic )
            options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( ui_shared_state.camera_perspective_fov );
        view.updateOptions( options );
        needs_redraw();
    }
    ui_shared_state.interaction_mode = interaction_mode;
}

} // namespace ui