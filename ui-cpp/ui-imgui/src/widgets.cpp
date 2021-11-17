#include <fonts.hpp>
#include <images.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Geometry.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <fmt/format.h>

#include <nfd.h>

#include <map>

namespace widgets
{

template<typename TYPE, typename FLOATTYPE>
float SliderCalcRatioFromValueT(
    ImGuiDataType data_type, TYPE v, TYPE v_min, TYPE v_max, float power, float linear_zero_pos )
{
    if( v_min == v_max )
        return 0.0f;

    const bool is_power
        = ( power != 1.0f ) && ( data_type == ImGuiDataType_Float || data_type == ImGuiDataType_Double );
    const TYPE v_clamped = ( v_min < v_max ) ? ImClamp( v, v_min, v_max ) : ImClamp( v, v_max, v_min );
    if( is_power )
    {
        if( v_clamped < 0.0f )
        {
            const float f
                = 1.0f
                  - static_cast<float>( ( v_clamped - v_min ) / ( ImMin( static_cast<TYPE>( 0 ), v_max ) - v_min ) );
            return ( 1.0f - ImPow( f, 1.0f / power ) ) * linear_zero_pos;
        }
        else
        {
            const float f = static_cast<float>(
                ( v_clamped - ImMax( static_cast<TYPE>( 0 ), v_min ) )
                / ( v_max - ImMax( static_cast<TYPE>( 0 ), v_min ) ) );
            return linear_zero_pos + ImPow( f, 1.0f / power ) * ( 1.0f - linear_zero_pos );
        }
    }

    // Linear slider
    return static_cast<float>( static_cast<FLOATTYPE>( v_clamped - v_min ) / static_cast<FLOATTYPE>( v_max - v_min ) );
}
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

void show_overlay_system(
    bool & show, int & corner, std::array<float, 2> & position, std::shared_ptr<State> state, ImVec2 viewport_pos,
    ImVec2 viewport_size )
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

    const float DISTANCE = 15.0f;

    static bool need_to_position = true;

    ImGuiIO & io = ImGui::GetIO();
    if( viewport_size[0] < 0 )
        viewport_size[0] = io.DisplaySize.x;
    if( viewport_size[1] < 0 )
        viewport_size[1] = io.DisplaySize.y;

    if( corner != -1 )
    {
        ImVec2 window_pos = viewport_pos
                            + ImVec2(
                                ( corner & 1 ) ? viewport_size[0] - DISTANCE : DISTANCE,
                                ( corner & 2 ) ? viewport_size[1] - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
        need_to_position = false;
    }
    else if( need_to_position )
    {
        ImGui::SetNextWindowPos( viewport_pos + ImVec2{ position[0], position[1] } );
        need_to_position = false;
    }

    ImGui::SetNextWindowBgAlpha( 0.6f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
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
    std::array<float, 2> & position, std::shared_ptr<State> state, ImVec2 viewport_pos, ImVec2 viewport_size )
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

    const float DISTANCE = 15.0f;

    ImGuiIO & io = ImGui::GetIO();
    if( viewport_size[0] < 0 )
        viewport_size[0] = io.DisplaySize.x;
    if( viewport_size[1] < 0 )
        viewport_size[1] = io.DisplaySize.y;

    if( corner != -1 )
    {
        ImVec2 window_pos = viewport_pos
                            + ImVec2(
                                ( corner & 1 ) ? viewport_size[0] - DISTANCE : DISTANCE,
                                ( corner & 2 ) ? viewport_size[1] - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
        need_to_position = false;
    }
    else if( need_to_position )
    {
        ImGui::SetNextWindowPos( viewport_pos + ImVec2{ position[0], position[1] } );
        need_to_position = false;
    }

    ImGui::SetNextWindowBgAlpha( 0.6f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
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

    auto & ui_state = rendering_layer.ui_shared_state;

    ImGui::Begin( "Settings", &show );

    ImGui::TextUnformatted( ICON_FA_SUN " light mode" );
    ImGui::SameLine();
    if( toggle_button( "##toggle_update_geometry_file_read", &ui_state.dark_mode, false ) )
    {
        if( ui_state.dark_mode )
        {
            styles::apply_charcoal();
            ui_state.dark_mode = true;
            rendering_layer.set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>(
                glm::vec3{ ui_state.background_dark[0], ui_state.background_dark[1], ui_state.background_dark[2] } );
        }
        else
        {
            styles::apply_light();
            ui_state.dark_mode = false;
            rendering_layer.set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>(
                glm::vec3{ ui_state.background_light[0], ui_state.background_light[1], ui_state.background_light[2] } );
        }
        rendering_layer.update_theme();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted( ICON_FA_MOON " dark mode" );

    ImGui::Dummy( { 0, 10 } );

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

    ImGui::Dummy( { 0, 10 } );

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
    ImGui::TextUnformatted( fmt::format( "     # of threads: {}", Spirit_OpenMP_Get_Num_Threads() ).c_str() );

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

float RoundScalarWithFormatFloat( const char * format, ImGuiDataType data_type, float v )
{
    return ImGui::RoundScalarWithFormatT<float, float>( format, data_type, v );
}

float SliderCalcRatioFromValueFloat(
    ImGuiDataType data_type, float v, float v_min, float v_max, float power, float linear_zero_pos )
{
    return SliderCalcRatioFromValueT<float, float>( data_type, v, v_min, v_max, power, linear_zero_pos );
}

bool RangeSliderBehavior(
    const ImRect & frame_bb, ImGuiID id, float * v1, float * v2, float v_min, float v_max, float power,
    int decimal_precision, ImGuiSliderFlags flags )
{
    ImGuiContext & g         = *GImGui;
    ImGuiWindow * window     = ImGui::GetCurrentWindow();
    const ImGuiStyle & style = g.Style;

    // Draw frame
    ImGui::RenderFrame( frame_bb.Min, frame_bb.Max, ImGui::GetColorU32( ImGuiCol_FrameBg ), true, style.FrameRounding );

    const bool is_non_linear = ( power < 1.0f - 0.00001f ) || ( power > 1.0f + 0.00001f );
    const bool is_horizontal = ( flags & ImGuiSliderFlags_Vertical ) == 0;

    const float grab_padding = 2.0f;
    const float slider_sz    = is_horizontal ? ( frame_bb.GetWidth() - grab_padding * 2.0f ) :
                                               ( frame_bb.GetHeight() - grab_padding * 2.0f );
    float grab_sz;
    if( decimal_precision > 0 )
        grab_sz = ImMin( style.GrabMinSize, slider_sz );
    else
        grab_sz = ImMin(
            ImMax(
                1.0f * ( slider_sz / ( ( v_min < v_max ? v_max - v_min : v_min - v_max ) + 1.0f ) ),
                style.GrabMinSize ),
            slider_sz ); // Integer sliders, if possible have the grab size represent 1 unit
    const float slider_usable_sz = slider_sz - grab_sz;
    const float slider_usable_pos_min
        = ( is_horizontal ? frame_bb.Min.x : frame_bb.Min.y ) + grab_padding + grab_sz * 0.5f;
    const float slider_usable_pos_max
        = ( is_horizontal ? frame_bb.Max.x : frame_bb.Max.y ) - grab_padding - grab_sz * 0.5f;

    // For logarithmic sliders that cross over sign boundary we want the exponential increase to be symmetric around 0.0f
    float linear_zero_pos = 0.0f; // 0.0->1.0f
    if( v_min * v_max < 0.0f )
    {
        // Different sign
        const float linear_dist_min_to_0 = powf( fabsf( 0.0f - v_min ), 1.0f / power );
        const float linear_dist_max_to_0 = powf( fabsf( v_max - 0.0f ), 1.0f / power );
        linear_zero_pos                  = linear_dist_min_to_0 / ( linear_dist_min_to_0 + linear_dist_max_to_0 );
    }
    else
    {
        // Same sign
        linear_zero_pos = v_min < 0.0f ? 1.0f : 0.0f;
    }

    // Process clicking on the slider
    bool value_changed = false;
    if( g.ActiveId == id )
    {
        if( g.IO.MouseDown[0] )
        {
            const float mouse_abs_pos = is_horizontal ? g.IO.MousePos.x : g.IO.MousePos.y;
            float clicked_t           = ( slider_usable_sz > 0.0f ) ?
                                            ImClamp( ( mouse_abs_pos - slider_usable_pos_min ) / slider_usable_sz, 0.0f, 1.0f ) :
                                            0.0f;
            if( !is_horizontal )
                clicked_t = 1.0f - clicked_t;

            float new_value;
            if( is_non_linear )
            {
                // Account for logarithmic scale on both sides of the zero
                if( clicked_t < linear_zero_pos )
                {
                    // Negative: rescale to the negative range before powering
                    float a   = 1.0f - ( clicked_t / linear_zero_pos );
                    a         = powf( a, power );
                    new_value = ImLerp( ImMin( v_max, 0.0f ), v_min, a );
                }
                else
                {
                    // Positive: rescale to the positive range before powering
                    float a;
                    if( fabsf( linear_zero_pos - 1.0f ) > 1.e-6f )
                        a = ( clicked_t - linear_zero_pos ) / ( 1.0f - linear_zero_pos );
                    else
                        a = clicked_t;
                    a         = powf( a, power );
                    new_value = ImLerp( ImMax( v_min, 0.0f ), v_max, a );
                }
            }
            else
            {
                // Linear slider
                new_value = ImLerp( v_min, v_max, clicked_t );
            }

            char fmt[64];
            snprintf( fmt, 64, "%%.%df", decimal_precision );

            // Round past decimal precision
            new_value = RoundScalarWithFormatFloat( fmt, ImGuiDataType_Float, new_value );
            if( *v1 != new_value || *v2 != new_value )
            {
                if( fabsf( *v1 - new_value ) < fabsf( *v2 - new_value ) )
                {
                    *v1 = new_value;
                }
                else
                {
                    *v2 = new_value;
                }
                value_changed = true;
            }
        }
        else
        {
            ImGui::ClearActiveID();
        }
    }

    // Calculate slider grab positioning
    float grab_t = SliderCalcRatioFromValueFloat( ImGuiDataType_Float, *v1, v_min, v_max, power, linear_zero_pos );

    // Draw
    if( !is_horizontal )
        grab_t = 1.0f - grab_t;
    float grab_pos = ImLerp( slider_usable_pos_min, slider_usable_pos_max, grab_t );
    ImRect grab_bb1;
    if( is_horizontal )
        grab_bb1 = ImRect(
            ImVec2( grab_pos - grab_sz * 0.5f, frame_bb.Min.y + grab_padding ),
            ImVec2( grab_pos + grab_sz * 0.5f, frame_bb.Max.y - grab_padding ) );
    else
        grab_bb1 = ImRect(
            ImVec2( frame_bb.Min.x + grab_padding, grab_pos - grab_sz * 0.5f ),
            ImVec2( frame_bb.Max.x - grab_padding, grab_pos + grab_sz * 0.5f ) );
    window->DrawList->AddRectFilled(
        grab_bb1.Min, grab_bb1.Max,
        ImGui::GetColorU32( g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab ), style.GrabRounding );

    // Calculate slider grab positioning
    grab_t = SliderCalcRatioFromValueFloat( ImGuiDataType_Float, *v2, v_min, v_max, power, linear_zero_pos );

    // Draw
    if( !is_horizontal )
        grab_t = 1.0f - grab_t;
    grab_pos = ImLerp( slider_usable_pos_min, slider_usable_pos_max, grab_t );
    ImRect grab_bb2;
    if( is_horizontal )
        grab_bb2 = ImRect(
            ImVec2( grab_pos - grab_sz * 0.5f, frame_bb.Min.y + grab_padding ),
            ImVec2( grab_pos + grab_sz * 0.5f, frame_bb.Max.y - grab_padding ) );
    else
        grab_bb2 = ImRect(
            ImVec2( frame_bb.Min.x + grab_padding, grab_pos - grab_sz * 0.5f ),
            ImVec2( frame_bb.Max.x - grab_padding, grab_pos + grab_sz * 0.5f ) );
    window->DrawList->AddRectFilled(
        grab_bb2.Min, grab_bb2.Max,
        ImGui::GetColorU32( g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab ), style.GrabRounding );

    ImRect connector( grab_bb1.Min, grab_bb2.Max );
    connector.Min.x += grab_sz;
    connector.Min.y += grab_sz * 0.3f;
    connector.Max.x -= grab_sz;
    connector.Max.y -= grab_sz * 0.3f;

    window->DrawList->AddRectFilled(
        connector.Min, connector.Max, ImGui::GetColorU32( ImGuiCol_SliderGrab ), style.GrabRounding );

    return value_changed;
}

// ~95% common code with ImGui::SliderFloat
bool RangeSliderFloat(
    const char * label, float * v1, float * v2, float v_min, float v_max, const char * display_format, float power )
{
    ImGuiWindow * window = ImGui::GetCurrentWindow();
    if( window->SkipItems )
        return false;

    ImGuiContext & g         = *GImGui;
    const ImGuiStyle & style = g.Style;
    const ImGuiID id         = window->GetID( label );
    const float w            = ImGui::CalcItemWidth();

    const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );
    const ImRect frame_bb(
        window->DC.CursorPos, window->DC.CursorPos + ImVec2( w, label_size.y + style.FramePadding.y * 2.0f ) );
    const ImRect total_bb(
        frame_bb.Min,
        frame_bb.Max + ImVec2( label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0.0f ) );

    // NB- we don't call ItemSize() yet because we may turn into a text edit box below
    if( !ImGui::ItemAdd( total_bb, id ) )
    {
        ImGui::ItemSize( total_bb, style.FramePadding.y );
        return false;
    }

    const bool hovered = ImGui::ItemHoverable( frame_bb, id );
    if( hovered )
        ImGui::SetHoveredID( id );

    if( !display_format )
        display_format = "(%.3f, %.3f)";
    int decimal_precision = ImParseFormatPrecision( display_format, 3 );

    // Tabbing or CTRL-clicking on Slider turns it into an input box
    bool start_text_input          = false;
    const bool tab_focus_requested = ImGui::FocusableItemRegister( window, g.ActiveId == id );
    if( tab_focus_requested || ( hovered && g.IO.MouseClicked[0] ) )
    {
        ImGui::SetActiveID( id, window );
        ImGui::FocusWindow( window );

        if( tab_focus_requested || g.IO.KeyCtrl )
        {
            start_text_input = true;
            g.TempInputId    = 0;
        }
    }

    if( start_text_input || ( g.ActiveId == id && g.TempInputId == id ) )
    {
        char fmt[64];
        snprintf( fmt, 64, "%%.%df", decimal_precision );
        return ImGui::TempInputScalar( frame_bb, id, label, ImGuiDataType_Float, v1, fmt );
    }

    ImGui::ItemSize( total_bb, style.FramePadding.y );

    // Actual slider behavior + render grab
    const bool value_changed = RangeSliderBehavior( frame_bb, id, v1, v2, v_min, v_max, power, decimal_precision, 0 );

    // Display value using user-provided display format so user can add prefix/suffix/decorations to the value.
    char value_buf[64];
    const char * value_buf_end
        = value_buf + ImFormatString( value_buf, IM_ARRAYSIZE( value_buf ), display_format, *v1, *v2 );
    ImGui::RenderTextClipped( frame_bb.Min, frame_bb.Max, value_buf, value_buf_end, NULL, ImVec2( 0.5f, 0.5f ) );

    if( label_size.x > 0.0f )
        ImGui::RenderText(
            ImVec2( frame_bb.Max.x + style.ItemInnerSpacing.x, frame_bb.Min.y + style.FramePadding.y ), label );

    return value_changed;
}

} // namespace widgets

template<typename TYPE>
static const char * ImAtoi( const char * src, TYPE * output )
{
    int negative = 0;
    if( *src == '-' )
    {
        negative = 1;
        src++;
    }
    if( *src == '+' )
    {
        src++;
    }
    TYPE v = 0;
    while( *src >= '0' && *src <= '9' )
        v = ( v * 10 ) + ( *src++ - '0' );
    *output = negative ? -v : v;
    return src;
}

template<typename TYPE, typename SIGNEDTYPE>
TYPE ImGui::RoundScalarWithFormatT( const char * format, ImGuiDataType data_type, TYPE v )
{
    const char * fmt_start = ImParseFormatFindStart( format );
    if( fmt_start[0] != '%' || fmt_start[1] == '%' ) // Don't apply if the value is not visible in the format string
        return v;
    char v_str[64];
    ImFormatString( v_str, IM_ARRAYSIZE( v_str ), fmt_start, v );
    const char * p = v_str;
    while( *p == ' ' )
        p++;
    if( data_type == ImGuiDataType_Float || data_type == ImGuiDataType_Double )
        v = (TYPE)ImAtof( p );
    else
        ImAtoi( p, (SIGNEDTYPE *)&v );
    return v;
}