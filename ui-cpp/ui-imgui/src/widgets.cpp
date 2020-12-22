#include <fonts.hpp>
#include <images.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <implot/implot.h>

#include <fmt/format.h>

#include <nfd.h>

#include <map>

namespace widgets
{

bool toggle_button( const char * str_id, bool * v, bool coloured )
{
    ImVec2 p               = ImGui::GetCursorScreenPos();
    ImDrawList * draw_list = ImGui::GetWindowDrawList();
    bool clicked           = false;

    float height = ImGui::GetFrameHeight() * 0.8f;
    float radius = height * 0.50f;
    float width  = 2 * radius * 1.55f;

    if( ImGui::InvisibleButton( str_id, ImVec2( width, height ) ) )
    {
        *v      = !*v;
        clicked = true;
    }
    ImU32 col_bg;
    if( ImGui::IsItemHovered() )
        col_bg
            = *v && coloured ? IM_COL32( 145 + 20, 211, 68 + 20, 255 ) : IM_COL32( 218 - 20, 218 - 20, 218 - 20, 255 );
    else
        col_bg = *v && coloured ? IM_COL32( 145, 211, 68, 255 ) : IM_COL32( 218, 218, 218, 255 );

    draw_list->AddRectFilled( p, ImVec2( p.x + width, p.y + height ), col_bg, height * 0.5f );
    draw_list->AddCircleFilled(
        ImVec2( *v ? ( p.x + width - radius ) : ( p.x + radius ), p.y + radius ), radius - 1.5f,
        IM_COL32( 255, 255, 255, 255 ) );

    return clicked;
}

void show_plots( bool & show, std::shared_ptr<State> state )
{
    auto & style = ImGui::GetStyle();

    static bool plot_image_energies        = true;
    static bool plot_interpolated_energies = true;

    static int history_size = 200;
    static std::vector<float> force_history( history_size );
    static std::vector<float> iteration_history( history_size );
    static int force_index = 0;
    static float force_min = 0;
    static float force_max = 0;

    static bool animate = true;

    // static float values[90]    = {};
    static int n_interpolate = Parameters_GNEB_Get_N_Energy_Interpolations( state.get() );
    static std::vector<float> rx( 1, 0 );
    static std::vector<float> energies( 1, 0 );
    static std::vector<float> rx_interpolated( 1, 0 );
    static std::vector<float> energies_interpolated( 1, 0 );

    if( Simulation_Running_Anywhere_On_Chain( state.get() ) )
    {
        float force = Simulation_Get_MaxTorqueNorm( state.get() );

        iteration_history[force_index] = Simulation_Get_Iteration( state.get() );

        force_history[force_index] = force;
        ++force_index;
        force_index = force_index % history_size;

        if( force < force_min )
            force_min = force;
        if( force > force_max )
            force_max = force;
    }

    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 500, 300 }, { 800, 999999 } );

    ImGui::Begin( "Plots", &show );
    {
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
        if( ImGui::BeginTabBar( "plots_tab_bar", tab_bar_flags ) )
        {
            if( ImGui::BeginTabItem( "Energy" ) )
            {
                int noi = Chain_Get_NOI( state.get() );
                if( noi != energies.size() )
                {
                    rx.resize( noi );
                    energies.resize( noi );
                }

                int size_interp = noi + ( noi - 1 ) * n_interpolate;
                if( size_interp != energies_interpolated.size() )
                {
                    rx_interpolated.resize( size_interp );
                    energies_interpolated.resize( size_interp );
                }

                Chain_Get_Rx( state.get(), rx.data() );
                Chain_Get_Energy( state.get(), energies.data() );

                if( noi > 1 )
                {
                    Chain_Get_Rx_Interpolated( state.get(), rx_interpolated.data() );
                    Chain_Get_Energy_Interpolated( state.get(), energies_interpolated.data() );
                }

                std::string overlay = fmt::format( "{:.3e}", energies[System_Get_Index( state.get() )] );

                // ImPlot::FitNextPlotAxes();
                if( ImPlot::BeginPlot(
                        "", "Rx", "E [meV]",
                        ImVec2(
                            ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                            ImGui::GetWindowContentRegionMax().y - 110.f ) ) )
                {
                    if( plot_image_energies )
                    {
                        ImPlot::PlotScatter( "", rx.data(), energies.data(), noi );
                    }
                    if( plot_interpolated_energies && noi > 1 )
                    {
                        ImPlot::PlotLine( "", rx_interpolated.data(), energies_interpolated.data(), size_interp );
                    }

                    ImPlot::EndPlot();
                }

                ImGui::Checkbox( "Image energies", &plot_image_energies );
                ImGui::Checkbox( "Interpolated energies", &plot_interpolated_energies );
                ImGui::SameLine();
                ImGui::PushItemWidth( 100 );
                if( ImGui::InputInt( "##energies_n_interp", &n_interpolate ) )
                {
                    if( n_interpolate > 1 )
                        Parameters_GNEB_Set_N_Energy_Interpolations( state.get(), n_interpolate );
                    else
                        n_interpolate = 1;
                }
                ImGui::PopItemWidth();

                ImGui::EndTabItem();
            }
            if( ImGui::BeginTabItem( "Convergence" ) )
            {
                ImPlot::FitNextPlotAxes();
                if( ImPlot::BeginPlot(
                        "", "iteration", "max(F)",
                        ImVec2(
                            ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                            ImGui::GetWindowContentRegionMax().y - 90.f ) ) )
                {
                    ImPlot::PlotLine( "", iteration_history.data(), force_history.data(), history_size, force_index );
                    ImPlot::EndPlot();
                }

                if( ImGui::InputInt( "history size", &history_size, 10, 100 ) )
                {
                    if( history_size < 3 )
                        history_size = 3;
                    iteration_history.resize( history_size );
                    force_history.resize( history_size );
                }

                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void show_configurations( bool & show, std::shared_ptr<State> state, ui::RenderingLayer & rendering_layer )
{
    static float pos[3]{ 0, 0, 0 };
    static float border_rect[3]{ -1, -1, -1 };
    static float border_cyl = -1;
    static float border_sph = -1;
    static bool inverted    = false;

    static float temperature = 0;

    static float sk_radius = 15;
    static float sk_speed  = 1;
    static float sk_phase  = 0;
    static bool sk_up_down = false;
    static bool sk_achiral = false;
    static bool sk_rl      = false;

    static float hopfion_radius = 10;
    static int hopfion_order    = 1;

    static float spiral_angle    = 0;
    static float spiral_axis[3]  = { 0, 0, 1 };
    static float spiral_qmag     = 1;
    static float spiral_qvec[3]  = { 1, 0, 0 };
    static bool spiral_q2        = false;
    static float spiral_qmag2    = 1;
    static float spiral_qvec2[3] = { 1, 0, 0 };

    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Configurations", &show );

    if( ImGui::CollapsingHeader( "Settings" ) )
    {
        ImGui::Indent( 15 );

        ImGui::TextUnformatted( "pos" );
        ImGui::SameLine();
        ImGui::InputFloat3( "##configurations_pos", pos );

        ImGui::TextUnformatted( "Rectangular border" );
        ImGui::SameLine();
        ImGui::InputFloat3( "##configurations_border_rect", border_rect );

        ImGui::TextUnformatted( "Cylindrical border" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 50 );
        ImGui::InputFloat( "##configurations_border_cyl", &border_cyl );

        ImGui::TextUnformatted( "Spherical border" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 50 );
        ImGui::InputFloat( "##configurations_border_sph", &border_sph );

        ImGui::TextUnformatted( "invert" );
        ImGui::SameLine();
        ImGui::Checkbox( "##configurations_inverted", &inverted );

        ImGui::Indent( -15 );
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Random" ) )
    {
        Configuration_Random( state.get() );
        rendering_layer.needs_data();
    }
    ImGui::SameLine();
    if( ImGui::Button( "+z" ) )
    {
        Configuration_PlusZ( state.get() );
        rendering_layer.needs_data();
    }
    ImGui::SameLine();
    if( ImGui::Button( "-z" ) )
    {
        Configuration_MinusZ( state.get() );
        rendering_layer.needs_data();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_noise", &temperature );
    ImGui::SameLine();
    if( ImGui::Button( "Add noise" ) )
    {
        Configuration_Add_Noise_Temperature(
            state.get(), temperature, pos, border_rect, border_cyl, border_sph, inverted );

        Chain_Update_Data( state.get() );
        rendering_layer.needs_data();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Skyrmion" ) )
    {

        Configuration_Skyrmion(
            state.get(), sk_radius, sk_speed, sk_phase, sk_up_down, sk_achiral, sk_rl, pos, border_rect, border_cyl,
            border_sph, inverted );

        rendering_layer.needs_data();
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "Radius" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_radius", &sk_radius );
    ImGui::TextUnformatted( "Speed" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_speed", &sk_speed );
    ImGui::TextUnformatted( "Phase" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_phase", &sk_phase );

    ImGui::TextUnformatted( "Up" );
    ImGui::SameLine();
    toggle_button( "##configurations_sk_up_down", &sk_up_down, false );
    ImGui::SameLine();
    ImGui::TextUnformatted( "Down" );

    ImGui::TextUnformatted( "Achiral" );
    ImGui::SameLine();
    ImGui::Checkbox( "##configurations_sk_achiral", &sk_achiral );

    ImGui::TextUnformatted( "Invert vorticity" );
    ImGui::SameLine();
    ImGui::Checkbox( "##configurations_sk_rl", &sk_rl );

    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Hopfion" ) )
    {
        Configuration_Hopfion(
            state.get(), hopfion_radius, hopfion_order, pos, border_rect, border_cyl, border_sph, inverted );

        rendering_layer.needs_data();
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "Radius" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 60 );
    ImGui::InputFloat( "##configurations_hopfion_radius", &hopfion_radius );

    ImGui::TextUnformatted( "Order" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 80 );
    ImGui::InputInt( "##configurations_hopfion_order", &hopfion_order );

    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    const char * direction_type = "Real Lattice";
    // if( comboBox_SS->currentText() == "Real Lattice" )
    //     direction_type = "Real Lattice";
    // else if( comboBox_SS->currentText() == "Reciprocal Lattice" )
    //     direction_type = "Reciprocal Lattice";
    // else if( comboBox_SS->currentText() == "Real Space" )
    //     direction_type = "Real Space";

    if( ImGui::Button( "Spiral" ) )
    {
        // Normalize
        float absq = std::sqrt(
            spiral_qvec[0] * spiral_qvec[0] + spiral_qvec[1] * spiral_qvec[1] + spiral_qvec[2] * spiral_qvec[2] );
        if( absq == 0 )
        {
            spiral_qvec[0] = 0;
            spiral_qvec[1] = 0;
            spiral_qvec[2] = 1;
        }

        // Scale
        for( int dim = 0; dim < 3; ++dim )
            spiral_qvec[dim] *= spiral_qmag;

        // Create Spin Spiral
        if( spiral_q2 )
        {
            // Normalize
            float absq2 = std::sqrt(
                spiral_qvec2[0] * spiral_qvec2[0] + spiral_qvec2[1] * spiral_qvec2[1]
                + spiral_qvec2[2] * spiral_qvec2[2] );
            if( absq == 0 )
            {
                spiral_qvec2[0] = 0;
                spiral_qvec2[1] = 0;
                spiral_qvec2[2] = 1;
            }

            // Scale
            for( int dim = 0; dim < 3; ++dim )
                spiral_qvec2[dim] *= spiral_qmag2;

            Configuration_SpinSpiral_2q(
                state.get(), direction_type, spiral_qvec, spiral_qvec2, spiral_axis, spiral_angle, pos, border_rect,
                border_cyl, border_sph, inverted );
        }
        else
            Configuration_SpinSpiral(
                state.get(), direction_type, spiral_qvec, spiral_axis, spiral_angle, pos, border_rect, border_cyl,
                border_sph, inverted );
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "q" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_spiral_qmag", &spiral_qmag );

    ImGui::TextUnformatted( "q dir" );
    ImGui::SameLine();
    ImGui::InputFloat3( "##configurations_spiral_qvec", spiral_qvec );

    ImGui::TextUnformatted( "axis" );
    ImGui::SameLine();
    ImGui::InputFloat3( "##configurations_spiral_axis", spiral_axis );

    ImGui::TextUnformatted( "angle" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_spiral_angle", &spiral_angle );

    ImGui::Indent( -15 );

    ImGui::End();
}

void show_parameters( bool & show, GUI_Mode & selected_mode )
{
    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Parameters", &show );

    if( selected_mode == GUI_Mode::Minimizer )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::MC )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::LLG )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::GNEB )
    {
    }
    else if( selected_mode == GUI_Mode::MMF )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::EMA )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }

    ImGui::End();
}

void show_geometry( bool & show, std::shared_ptr<State> state, ui::RenderingLayer & rendering_layer )
{
    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Geometry", &show );

    int n_cells[3];
    Geometry_Get_N_Cells( state.get(), n_cells );
    ImGui::TextUnformatted( "Number of cells" );
    ImGui::SameLine();
    if( ImGui::InputInt3( "##geometry_n_cells", n_cells, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
        Geometry_Set_N_Cells( state.get(), n_cells );
        rendering_layer.update_vf_geometry();
    }

    ImGui::End();
}

void show_visualisation_widget( bool & show, ui::RenderingLayer & rendering_layer )
{
    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Visualisation settings", &show );

    float * colour = rendering_layer.ui_shared_state.background_light.data();
    if( rendering_layer.ui_shared_state.dark_mode )
        colour = rendering_layer.ui_shared_state.background_dark.data();

    ImGui::TextUnformatted( "Background color" );
    ImGui::SameLine();
    if( ImGui::Button( "default" ) )
    {
        if( rendering_layer.ui_shared_state.dark_mode )
            rendering_layer.ui_shared_state.background_dark = { 0.4f, 0.4f, 0.4f };
        else
            rendering_layer.ui_shared_state.background_light = { 0.9f, 0.9f, 0.9f };

        rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { colour[0], colour[1], colour[2] } );
    }

    if( ImGui::ColorEdit3( "##bgcolour", colour ) )
    {
        rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { colour[0], colour[1], colour[2] } );
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    ImGui::TextUnformatted( "Renderers" );
    if( ImGui::Button( "Add Renderer" ) )
    {
        if( ImGui::IsPopupOpen( "##popup_add_renderer" ) )
            ImGui::CloseCurrentPopup();
        else
        {
            ImGui::OpenPopup( "##popup_add_renderer" );
        }
    }

    if( ImGui::BeginPopup( "##popup_add_renderer" ) )
    {
        std::shared_ptr<ui::RendererWidget> renderer;
        if( ImGui::Selectable( "Dots" ) )
        {
            renderer = std::make_shared<ui::DotRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Arrows" ) )
        {
            renderer = std::make_shared<ui::ArrowRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Boxes" ) )
        {
            renderer = std::make_shared<ui::ParallelepipedRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Spheres" ) )
        {
            renderer = std::make_shared<ui::SphereRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Surface" ) )
        {
            renderer = std::make_shared<ui::SurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Isosurface" ) )
        {
            renderer = std::make_shared<ui::IsosurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( renderer )
        {
            rendering_layer.renderer_widgets.push_back( renderer );
            rendering_layer.renderer_widgets_not_shown.push_back( renderer );
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    for( auto & renderer_widget : rendering_layer.renderer_widgets )
    {
        ImGui::Dummy( { 0, 10 } );
        ImGui::Separator();
        ImGui::Dummy( { 0, 10 } );

        renderer_widget->show();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    vgm::Vec3 dir(
        rendering_layer.ui_shared_state.light_direction[0], rendering_layer.ui_shared_state.light_direction[1],
        rendering_layer.ui_shared_state.light_direction[2] );
    bool update = false;
    ImGui::TextUnformatted( "Light direction" );
    ImGui::Columns( 2, "lightdircolumns", false ); // 3-ways, no border
    if( ImGui::gizmo3D( "##dir", dir ) )
        update = true;
    ImGui::NextColumn();
    auto normalize_light_dir = [&]() {
        auto norm = std::sqrt( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    };
    if( ImGui::InputFloat( "##lightdir_x", &dir.x, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_y", &dir.y, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_z", &dir.z, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( update )
    {
        normalize_light_dir();
        rendering_layer.ui_shared_state.light_direction = { dir.x, dir.y, dir.z };
        rendering_layer.view.setOption<VFRendering::View::Option::LIGHT_POSITION>(
            { -1000 * dir.x, -1000 * dir.y, -1000 * dir.z } );
    }

    ImGui::Columns( 1 );

    ImGui::End();
}

void show_overlay_system( bool & show, int & corner, std::array<float, 2> & position, std::shared_ptr<State> state )
{
    if( !show )
        return;

    static float energy = 0;
    static float m_x    = 0;
    static float m_y    = 0;
    static float m_z    = 0;

    static int noi           = 1;
    static int nos           = 1;
    static int n_basis_atoms = 1;
    static int n_cells[3]{ 1, 1, 1 };

    const float DISTANCE = 50.0f;

    static bool need_to_position = true;

    ImGuiIO & io = ImGui::GetIO();

    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
        need_to_position = false;
    }
    else if( need_to_position )
    {
        ImGui::SetNextWindowPos( { position[0], position[1] } );
        need_to_position = false;
    }

    ImGui::SetNextWindowBgAlpha( 0.45f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;
    if( ImGui::Begin( "System information overlay", &show, window_flags ) )
    {
        if( corner == -1 )
        {
            auto im_pos = ImGui::GetWindowPos();
            position[0] = im_pos.x;
            position[1] = im_pos.y;
        }

        noi = Chain_Get_NOI( state.get() );

        nos    = Geometry_Get_NOS( state.get() );
        energy = System_Get_Energy( state.get() );
        Geometry_Get_N_Cells( state.get(), n_cells );
        n_basis_atoms = Geometry_Get_N_Cell_Atoms( state.get() );

        ImGui::TextUnformatted( fmt::format( "FPS: {:d}", int( io.Framerate ) ).c_str() );

        ImGui::Separator();

        ImGui::TextUnformatted( fmt::format( "E      = {:.10f}", energy ).c_str() );
        ImGui::TextUnformatted( fmt::format( "E dens = {:.10f}", energy / nos ).c_str() );

        ImGui::Separator();

        ImGui::TextUnformatted( fmt::format( "M_x = {:.8f}", m_x ).c_str() );
        ImGui::TextUnformatted( fmt::format( "M_y = {:.8f}", m_y ).c_str() );
        ImGui::TextUnformatted( fmt::format( "M_z = {:.8f}", m_z ).c_str() );

        ImGui::Separator();

        ImGui::TextUnformatted( fmt::format( "NOI: {}", noi ).c_str() );
        ImGui::TextUnformatted( fmt::format( "NOS: {}", nos ).c_str() );
        ImGui::TextUnformatted( fmt::format( "N basis atoms: {}", n_basis_atoms ).c_str() );
        ImGui::TextUnformatted( fmt::format( "Cells: {}x{}x{}", n_cells[0], n_cells[1], n_cells[2] ).c_str() );

        ImGui::Separator();

        ImGui::TextUnformatted( "Simple overlay\n"
                                "in the corner of the screen.\n"
                                "(right-click to change position)" );

        ImGui::Separator();

        if( ImGui::IsMousePosValid() )
            ImGui::Text( "Mouse Position: (%.1f,%.1f)", io.MousePos.x, io.MousePos.y );
        else
            ImGui::TextUnformatted( "Mouse Position: <invalid>" );

        if( ImGui::BeginPopupContextWindow() )
        {
            if( ImGui::MenuItem( "Custom", NULL, corner == -1 ) )
                corner = -1;
            if( ImGui::MenuItem( "Top-left", NULL, corner == 0 ) )
                corner = 0;
            if( ImGui::MenuItem( "Top-right", NULL, corner == 1 ) )
                corner = 1;
            if( ImGui::MenuItem( "Bottom-left", NULL, corner == 2 ) )
                corner = 2;
            if( ImGui::MenuItem( "Bottom-right", NULL, corner == 3 ) )
                corner = 3;
            if( show && ImGui::MenuItem( "Close" ) )
                show = false;
            ImGui::EndPopup();
        }
        ImGui::End();
    }
}

void show_overlay_calculation(
    bool & show, GUI_Mode & selected_mode, int & selected_solver_min, int & selected_solver_llg, int & corner,
    std::array<float, 2> & position, std::shared_ptr<State> state )
{
    static bool need_to_position = true;

    if( !show )
        return;

    static auto solvers_llg = std::map<int, std::pair<std::string, std::string>>{
        { Solver_SIB, { "SIB", "Semi-implicit method B (Heun using approximated exponential transforms)" } },
        { Solver_Depondt, { "Depondt", "Depondt (Heun using rotations)" } },
        { Solver_Heun,
          { "Heun", "Heun's midpoint method, corresponding to RK2 (using cartesian finite differences)" } },
        { Solver_RungeKutta4, { "RK4", "4th order Runge-Kutta (using cartesian finite differences)" } }
    };

    static auto solvers_min = std::map<int, std::pair<std::string, std::string>>{
        { Solver_VP, { "VP", "Verlet-like velocity projection (using cartesian finite differences)" } },
        { Solver_VP_OSO, { "VP (OSO)", "Verlet-like velocity projection (using exponential transformations)" } },
        { Solver_LBFGS_OSO, { "LBFGS (OSO)", "LBFGS (using exponential transformations)" } },
        { Solver_LBFGS_Atlas, { "LBFGS (Atlas)", "LBFGS (using an atlas of coordinate maps)" } },
        { Solver_SIB, { "SIB", "Semi-implicit method B (Heun using approximated exponential transforms)" } },
        { Solver_Depondt, { "Depondt", "Depondt (Heun using rotations)" } },
        { Solver_Heun,
          { "Heun", "Heun's midpoint method, corresponding to RK2 (using cartesian finite differences)" } },
        { Solver_RungeKutta4, { "RK4", "4th order Runge-Kutta (using cartesian finite differences)" } }
    };

    static float solver_button_hovered_duration = 0;

    static float simulated_time = 0;
    static float wall_time      = 0;
    static int iteration        = 0;
    static float ips            = 0;

    int hours        = wall_time / ( 60 * 60 * 1000 );
    int minutes      = ( wall_time - 60 * 60 * 1000 * hours ) / ( 60 * 1000 );
    int seconds      = ( wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes ) / 1000;
    int milliseconds = wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes - 1000 * seconds;

    static float force_max = 0;

    // ips = Simulation_Get_IterationsPerSecond( state.get() );

    const float DISTANCE = 50.0f;

    ImGuiIO & io = ImGui::GetIO();
    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
        need_to_position = false;
    }
    else if( need_to_position )
    {
        ImGui::SetNextWindowPos( { position[0], position[1] } );
        need_to_position = false;
    }

    ImGui::SetNextWindowBgAlpha( 0.35f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;

    if( ImGui::Begin( "Calculation information overlay", &show, window_flags ) )
    {
        if( corner == -1 )
        {
            auto im_pos = ImGui::GetWindowPos();
            position[0] = im_pos.x;
            position[1] = im_pos.y;
        }

        if( Simulation_Running_Anywhere_On_Chain( state.get() ) )
        {
            simulated_time = Simulation_Get_Time( state.get() );
            wall_time      = Simulation_Get_Wall_Time( state.get() );
            iteration      = Simulation_Get_Iteration( state.get() );
            ips            = Simulation_Get_IterationsPerSecond( state.get() );

            force_max = Simulation_Get_MaxTorqueNorm( state.get() );
        }

        if( selected_mode == GUI_Mode::Minimizer || selected_mode == GUI_Mode::GNEB || selected_mode == GUI_Mode::MMF )
        {
            bool open_popup
                = ImGui::Button( fmt::format( "Solver: {}", solvers_min[selected_solver_min].first ).c_str() );

            if( ImGui::IsItemHovered() )
            {
                // 1.5s delay before showing tooltip
                solver_button_hovered_duration += io.DeltaTime;
                if( solver_button_hovered_duration > 1.5f )
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted( solvers_min[selected_solver_min].second.c_str() );
                    ImGui::EndTooltip();
                }
            }
            else
            {
                solver_button_hovered_duration = 0;
            }

            if( open_popup )
                ImGui::OpenPopup( "solver_popup_min" );
            if( ImGui::BeginPopup( "solver_popup_min" ) )
            {
                for( auto solver : solvers_min )
                    if( ImGui::Selectable( solver.second.first.c_str() ) )
                        selected_solver_min = solver.first;
                ImGui::EndPopup();
            }
            ImGui::Separator();
        }
        else if( selected_mode == GUI_Mode::LLG )
        {
            bool open_popup
                = ImGui::Button( fmt::format( "Solver: {}", solvers_llg[selected_solver_llg].first ).c_str() );

            if( ImGui::IsItemHovered() )
            {
                // 1.5s delay before showing tooltip
                solver_button_hovered_duration += io.DeltaTime;
                if( solver_button_hovered_duration > 1.5f )
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted( solvers_llg[selected_solver_llg].second.c_str() );
                    ImGui::EndTooltip();
                }
            }
            else
            {
                solver_button_hovered_duration = 0;
            }

            if( open_popup )
                ImGui::OpenPopup( "solver_popup_llg" );
            if( ImGui::BeginPopup( "solver_popup_llg" ) )
            {
                for( auto solver : solvers_llg )
                    if( ImGui::Selectable( solver.second.first.c_str() ) )
                        selected_solver_llg = solver.first;
                ImGui::EndPopup();
            }
            ImGui::Separator();
            ImGui::TextUnformatted( fmt::format( "t = {} ps", simulated_time ).c_str() );
        }

        ImGui::TextUnformatted(
            fmt::format( "{:0>2d}:{:0>2d}:{:0>2d}.{:0>3d}", hours, minutes, seconds, milliseconds ).c_str() );
        ImGui::TextUnformatted( fmt::format( "Iteration: {}", iteration ).c_str() );
        if( ips > 0.01 )
            ImGui::TextUnformatted( fmt::format( "IPS: {:>6.2f}", ips ).c_str() );
        else
            ImGui::TextUnformatted( fmt::format( "IPS: {}", ips ).c_str() );

        ImGui::Separator();

        ImGui::TextUnformatted( fmt::format( "F_max = {:.5e}", force_max ).c_str() );
        if( selected_mode == GUI_Mode::GNEB )
        {
            ImGui::TextUnformatted( fmt::format( "F_current = {:.5e}", simulated_time ).c_str() );
        }

        if( ImGui::BeginPopupContextWindow() )
        {
            if( ImGui::MenuItem( "Custom", NULL, corner == -1 ) )
                corner = -1;
            if( ImGui::MenuItem( "Top-left", NULL, corner == 0 ) )
                corner = 0;
            if( ImGui::MenuItem( "Top-right", NULL, corner == 1 ) )
                corner = 1;
            if( ImGui::MenuItem( "Bottom-left", NULL, corner == 2 ) )
                corner = 2;
            if( ImGui::MenuItem( "Bottom-right", NULL, corner == 3 ) )
                corner = 3;
            if( show && ImGui::MenuItem( "Close" ) )
                show = false;
            ImGui::EndPopup();
        }
        ImGui::End();
    }
}

void show_settings( bool & show, ui::RenderingLayer & rendering_layer )
{
    if( !show )
        return;

    ImGui::Begin( "Settings", &show );

    if( rendering_layer.ui_shared_state.dark_mode )
    {
        if( ImGui::Button( ICON_FA_SUN " light mode" ) )
        {
            styles::apply_light();
            rendering_layer.ui_shared_state.dark_mode = false;
            rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                { rendering_layer.ui_shared_state.background_light[0],
                  rendering_layer.ui_shared_state.background_light[1],
                  rendering_layer.ui_shared_state.background_light[2] } );
        }
    }
    else
    {
        if( ImGui::Button( ICON_FA_MOON " dark mode" ) )
        {
            styles::apply_charcoal();
            rendering_layer.ui_shared_state.dark_mode = true;
            rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                { rendering_layer.ui_shared_state.background_dark[0],
                  rendering_layer.ui_shared_state.background_dark[1],
                  rendering_layer.ui_shared_state.background_dark[2] } );
        }
    }

    if( ImGui::Button( "Choose the output folder" ) )
    {
        nfdchar_t * outPath = NULL;
        nfdresult_t result  = NFD_PickFolder( NULL, &outPath );
        if( result == NFD_OKAY )
        {
            fmt::print( "Folder path: \"{}\"\n", outPath );
            free( outPath );
        }
        else if( result == NFD_CANCEL )
        {
            fmt::print( "User pressed cancel.\n" );
        }
        else
        {
            fmt::print( "Error: {}\n", NFD_GetError() );
        }
    }

    if( ImGui::Button( "Close" ) )
        show = false;

    ImGui::End();
}

void show_keybindings( bool & show )
{
    if( !show )
        return;

    ImGui::SetNextWindowSizeConstraints( { 500, 300 }, { 800, 800 } );

    ImGui::Begin( "Keybindings", &show );

    ImGui::TextUnformatted( "UI controls" );
    ImGui::BulletText( "F1: Show this" );
    ImGui::BulletText( "F2: Toggle settings" );
    ImGui::BulletText( "F3: Toggle plots" );
    ImGui::BulletText( "F4: Toggle debug" );
    ImGui::BulletText( "F5: Toggle \"Dragging\" mode" );
    ImGui::BulletText( "F6: Toggle \"Defects\" mode" );
    ImGui::BulletText( "F7: Toggle \"Pinning\" mode" );
    ImGui::TextUnformatted( "" );
    ImGui::BulletText( "F10  and Ctrl+F:      Toggle large visualisation" );
    ImGui::BulletText( "F11 and Ctrl+Shift+F: Toggle fullscreen window" );
    ImGui::BulletText( "F12 and Home:         Screenshot of visualisation region" );
    ImGui::BulletText( "Ctrl+Shift+V:         Toggle OpenGL visualisation" );
    ImGui::BulletText( "i:                    Toggle large visualisation" );
    ImGui::BulletText( "Escape:               Try to return focus to main UI (does not always work)" );
    ImGui::TextUnformatted( "" );
    ImGui::TextUnformatted( "Camera controls" );
    ImGui::BulletText( "Left mouse:   Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Right mouse:  Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Scroll mouse: Zoom in on focus point (<b>shift</b> to go slow)" );
    ImGui::BulletText( "WASD:         Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "TFGH:         Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "X,Y,Z:        Set the camera in X, Y or Z direction (<b>shift</b> to invert)" );
    ImGui::TextUnformatted( "" );
    ImGui::TextUnformatted( "Control Simulations" );
    ImGui::BulletText( "Space:        Start/stop calculation" );
    ImGui::BulletText( "Ctrl+M:       Cycle method" );
    ImGui::BulletText( "Ctrl+S:       Cycle solver" );
    ImGui::TextUnformatted( "" );
    ImGui::TextUnformatted( "Manipulate the current images" );
    ImGui::BulletText( "Ctrl+R:       Random configuration" );
    ImGui::BulletText( "Ctrl+N:       Add tempered noise" );
    ImGui::BulletText( "Enter:        Insert last used configuration" );
    ImGui::TextUnformatted( "" );
    ImGui::TextUnformatted( "Visualisation" );
    ImGui::BulletText( "+/-:          Use more/fewer data points of the vector field" );
    ImGui::BulletText( "1:            Regular Visualisation Mode" );
    ImGui::BulletText( "2:            Isosurface Visualisation Mode" );
    ImGui::BulletText( "3-5:          Slab (X,Y,Z) Visualisation Mode" );
    ImGui::BulletText( "/:            Cycle Visualisation Mode" );
    ImGui::BulletText( ", and .:      Move Slab (<b>shift</b> to go faster)" );
    ImGui::TextUnformatted( "" );
    ImGui::TextUnformatted( "Manipulate the chain of images" );
    ImGui::BulletText( "Arrows:          Switch between images and chains" );
    ImGui::BulletText( "Ctrl+X:          Cut   image" );
    ImGui::BulletText( "Ctrl+C:          Copy  image" );
    ImGui::BulletText( "Ctrl+V:          Paste image at current index" );
    ImGui::BulletText( "Ctrl+Left/Right: Insert left/right of current index<" );
    ImGui::BulletText( "Del:             Delete image" );
    ImGui::TextUnformatted( "" );
    ImGui::TextWrapped( "Note that some of the keybindings may only work correctly on US keyboard layout.\n"
                        "\n"
                        "For more information see the documentation at spirit-docs.readthedocs.io" );
    ImGui::TextUnformatted( "" );

    if( ImGui::Button( "Close" ) )
        show = false;

    ImGui::End();
}

void show_about( bool & show_about )
{
    static bool logo_loaded          = false;
    static int logo_width            = 0;
    static int logo_height           = 0;
    static unsigned int logo_texture = 0;

    if( !show_about )
        return;

    auto & style = ImGui::GetStyle();
    ImVec2 spacing{ 2 * style.FramePadding.x, 2 * style.FramePadding.y };

    if( !logo_loaded )
    {
        images::Image logo( "res/Logo_Ghost.png" );
        if( logo.image_data )
        {
            logo.get_gl_texture( logo_texture );
            logo_width  = logo.width;
            logo_height = logo.height;
            logo_loaded = true;
        }
    }

    ImGui::Begin( fmt::format( "About Spirit {}", Spirit_Version() ).c_str(), &show_about );

    int scaled_width  = ImGui::GetContentRegionAvailWidth() * 0.8;
    int scaled_height = logo_height * scaled_width / logo_width;
    ImGui::SameLine( ImGui::GetContentRegionAvailWidth() * 0.1, 0 );
    ImGui::Image( (void *)(intptr_t)logo_texture, ImVec2( scaled_width, scaled_height ) );

    ImGui::TextWrapped( "The Spirit GUI application incorporates intuitive visualisations,"
                        " powerful energy minimization, Monte Carlo, spin dynamics and"
                        " nudged elastic band calculation tools into a cross-platform user"
                        " interface." );

    ImGui::Dummy( spacing );
    ImGui::Separator();
    ImGui::Dummy( spacing );

    ImGui::TextWrapped( "Current maintainer:" );
    ImGui::TextWrapped( "Moritz Sallermann (m.sallermann@fz-juelich.de)" );

    ImGui::Dummy( spacing );

    ImGui::TextWrapped( "For more information, visit http://juspin.de" );

    ImGui::Dummy( spacing );

    ImGui::TextWrapped( "The sources are hosted at spirit-code.github.io"
                        " and the documentation can be found at https://spirit-docs.readthedocs.io" );

    ImGui::Dummy( spacing );
    ImGui::Separator();
    ImGui::Dummy( spacing );

    ImGui::TextUnformatted( fmt::format( "Full library version {}", Spirit_Version_Full() ).c_str() );
    ImGui::TextUnformatted( fmt::format( "Built with {}", Spirit_Compiler_Full() ).c_str() );

    ImGui::Dummy( spacing );

    ImGui::TextUnformatted( fmt::format( "Floating point precision = {}", Spirit_Scalar_Type() ).c_str() );

    ImGui::Dummy( spacing );

    ImGui::Columns( 2, "aboutinfocolumns", false );
    ImGui::TextUnformatted( "Parallelisation:" );
    ImGui::TextUnformatted( fmt::format( "   - OpenMP  = {}", Spirit_OpenMP() ).c_str() );
    ImGui::TextUnformatted( fmt::format( "   - Cuda    = {}", Spirit_Cuda() ).c_str() );
    ImGui::TextUnformatted( fmt::format( "   - Threads = {}", Spirit_Threads() ).c_str() );
    ImGui::NextColumn();
    ImGui::TextUnformatted( "Other:" );
    ImGui::TextUnformatted( fmt::format( "   - Defects = {}", Spirit_Defects() ).c_str() );
    ImGui::TextUnformatted( fmt::format( "   - Pinning = {}", Spirit_Pinning() ).c_str() );
    ImGui::TextUnformatted( fmt::format( "   - FFTW    = {}", Spirit_FFTW() ).c_str() );
    ImGui::Columns( 1 );

    ImGui::Dummy( spacing );

    if( ImGui::Button( "Close" ) )
        show_about = false;
    ImGui::End();
}

// Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.txt)
void help_marker( const char * description )
{
    ImGui::TextDisabled( "(?)" );
    if( ImGui::IsItemHovered() )
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos( ImGui::GetFontSize() * 35.0f );
        ImGui::TextUnformatted( description );
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace widgets