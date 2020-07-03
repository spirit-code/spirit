#include <fonts.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <fmt/format.h>

#include <nfd.h>

#include <map>
#include <string>
#include <thread>

namespace widgets
{

void show_menu_bar(
    GLFWwindow * window, ImFont * font, bool & dark_mode, ImVec4 & background_colour, GUI_Mode & selected_mode,
    int & selected_solver, VFRendering::View & vfr_view, bool & show_keybindings, bool & show_about,
    std::shared_ptr<State> state, std::vector<std::thread> & threads_image, std::thread & thread_chain )
{
    static auto modes = std::map<GUI_Mode, std::pair<std::string, std::string>>{
        { GUI_Mode::Minimizer, { "Minimizer", "(1) energy minimisation" } },
        { GUI_Mode::MC, { "Monte Carlo", "(2) Monte Carlo Stochastical sampling" } },
        { GUI_Mode::LLG, { "LLG", "(3) Landau-Lifshitz-Gilbert dynamics" } },
        { GUI_Mode::GNEB, { "GNEB", "(4) geodesic nudged elastic band calculation" } },
        { GUI_Mode::MMF, { "MMF", "(5) minimum mode following saddle point search" } },
        { GUI_Mode::EMA, { "Eigenmodes", "(6) eigenmode calculation and visualisation" } }
    };

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 7.f, 7.f ) );
    ImGui::PushFont( font );

    if( ImGui::BeginMainMenuBar() )
    {
        // mainmenu_height = ImGui::GetWindowSize().y;
        ImGui::PopStyleVar();

        // ImGui::SameLine();
        if( ImGui::BeginMenu( "File" ) )
        {
            if( ImGui::MenuItem( "Load cfg" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "cfg", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save current cfg" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "cfg", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load spin configuration" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save spin configuration" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "ovf;txt;csv", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Load system eigenmodes" ) )
            {
            }
            if( ImGui::MenuItem( "Save system eigenmodes" ) )
            {
            }
            if( ImGui::MenuItem( "Save energy per spin" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load chain" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save chain" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "ovf;txt;csv", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save energies" ) )
            {
            }
            if( ImGui::MenuItem( "Save interpolated energies" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Choose output folder" ) )
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
            ImGui::Separator();
            if( ImGui::MenuItem( "Take Screenshot" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Edit" ) )
        {
            if( ImGui::MenuItem( "Cut system", "ctrl+x" ) )
            {
            }
            if( ImGui::MenuItem( "Copy system", "ctrl+c" ) )
            {
            }
            if( ImGui::MenuItem( "Paste system", "ctrl+v" ) )
            {
            }
            if( ImGui::MenuItem( "Insert left", "ctrl+leftarrow" ) )
            {
            }
            if( ImGui::MenuItem( "Insert right", "ctrl+rightarrow" ) )
            {
            }
            if( ImGui::MenuItem( "Delete system", "del" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Controls" ) )
        {
            if( ImGui::MenuItem( "Start/stop calculation", "space" ) )
            {
            }
            if( ImGui::MenuItem( "Randomize spins", "ctrl+r" ) )
            {
            }
            if( ImGui::MenuItem( "Cycle method", "ctrl+m" ) )
            {
            }
            if( ImGui::MenuItem( "Cycle solver", "ctrl+s" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle dragging mode", "F5" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle defect mode", "F6" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle pinning mode", "F7" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "View" ) )
        {
            if( ImGui::MenuItem( "Toggle info-widgets", "i" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle settings" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle plots" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle geometry" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle debug" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Regular mode" ) )
            {
            }
            if( ImGui::MenuItem( "Isosurface mode" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode X" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Y" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Z" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle camera projection", "c" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle visualisation", "ctrl+f" ) )
            {
            }
            if( ImGui::MenuItem( "Fullscreen", "ctrl+shift+f" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Help" ) )
        {
            if( ImGui::MenuItem( "Keybindings", "F1" ) )
            {
                show_keybindings = true;
            }
            if( ImGui::MenuItem( "About" ) )
            {
                show_about = true;
            }
            ImGui::EndMenu();
        }

        float menu_end = ImGui::GetCursorPosX();

        auto io            = ImGui::GetIO();
        auto & style       = ImGui::GetStyle();
        float font_size_px = font->FontSize;
        float right_edge   = ImGui::GetWindowContentRegionMax().x;
        float bar_height   = ImGui::GetWindowContentRegionMax().y + 2 * style.FramePadding.y;
        float width;

        ImGui::PushStyleVar( ImGuiStyleVar_SelectableTextAlign, ImVec2( .5f, .5f ) );

        width = 2.5f * font_size_px;
        ImGui::SameLine( right_edge - width, 0 );
        if( dark_mode )
        {
            if( ImGui::Button( ICON_FA_SUN, ImVec2( width, bar_height ) ) )
            {
                ImGui::StyleColorsLight();
                background_colour = ImVec4( 0.9f, 0.9f, 0.9f, 1.f );
                dark_mode         = false;
                vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                    { background_colour.x, background_colour.y, background_colour.z } );
            }
        }
        else
        {
            if( ImGui::Button( ICON_FA_MOON, ImVec2( width, bar_height ) ) )
            {
                styles::apply_charcoal();
                background_colour = ImVec4( 0.4f, 0.4f, 0.4f, 1.f );
                dark_mode         = true;
                vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                    { background_colour.x, background_colour.y, background_colour.z } );
            }
        }
        right_edge -= ( width + style.FramePadding.x );

        // TODO: deactivate method selection if a calculation is running
        for( int n = modes.size(); n > 0; n-- )
        {
            auto mode         = GUI_Mode( n );
            std::string label = modes[mode].first;
            width             = label.length() * font_size_px * 0.6;
            ImGui::SameLine( right_edge - width, 0 );
            if( ImGui::Selectable( label.c_str(), selected_mode == mode, 0, ImVec2( width, bar_height ) ) )
                selected_mode = mode;

            if( ImGui::IsItemHovered() )
            {
                ImGui::BeginTooltip();
                ImGui::Text( modes[mode].second.c_str() );
                ImGui::EndTooltip();
            }
            right_edge -= ( width + 2 * style.FramePadding.x );
        }

        width             = 2.5f * font_size_px;
        float total_width = 3 * width + 40 + 4 * style.FramePadding.x;
        float start       = menu_end + 0.5f * ( right_edge - menu_end - total_width );

        ImGui::SameLine( start, 0 );
        bool calculation_running
            = Simulation_Running_On_Chain( state.get() ) || Simulation_Running_On_Image( state.get() );
        if( calculation_running )
        {
            if( ImGui::Button( ICON_FA_STOP_CIRCLE, ImVec2( width, bar_height ) ) )
            {
                // Running, so we stop it
                Simulation_Stop( state.get() );
                // Join the thread of the stopped simulation
                if( threads_image[System_Get_Index( state.get() )].joinable() )
                    threads_image[System_Get_Index( state.get() )].join();
                else if( thread_chain.joinable() )
                    thread_chain.join();
                // this->spinWidget->updateData();
            }
        }
        else
        {
            if( ImGui::Button( ICON_FA_PLAY_CIRCLE, ImVec2( width, bar_height ) ) )
            {
                // Not running, so we start it
                if( selected_mode == GUI_Mode::Minimizer )
                {
                    int idx = System_Get_Index( state.get() );
                    if( threads_image[idx].joinable() )
                        threads_image[System_Get_Index( state.get() )].join();
                    threads_image[System_Get_Index( state.get() )]
                        = std::thread( &Simulation_LLG_Start, state.get(), selected_solver, -1, -1, false, -1, -1 );
                }
                else if( selected_mode == GUI_Mode::LLG )
                {
                    int idx = System_Get_Index( state.get() );
                    if( threads_image[idx].joinable() )
                        threads_image[System_Get_Index( state.get() )].join();
                    threads_image[System_Get_Index( state.get() )]
                        = std::thread( &Simulation_LLG_Start, state.get(), selected_solver, -1, -1, false, -1, -1 );
                }
                else if( selected_mode == GUI_Mode::MC )
                {
                    int idx = System_Get_Index( state.get() );
                    if( threads_image[idx].joinable() )
                        threads_image[System_Get_Index( state.get() )].join();
                    threads_image[System_Get_Index( state.get() )]
                        = std::thread( &Simulation_MC_Start, state.get(), -1, -1, false, -1, -1 );
                }
                else if( selected_mode == GUI_Mode::GNEB )
                {
                    if( thread_chain.joinable() )
                        thread_chain.join();
                    thread_chain
                        = std::thread( &Simulation_GNEB_Start, state.get(), selected_solver, -1, -1, false, -1 );
                }
                else if( selected_mode == GUI_Mode::MMF )
                {
                    int idx = System_Get_Index( state.get() );
                    if( threads_image[idx].joinable() )
                        threads_image[System_Get_Index( state.get() )].join();
                    threads_image[System_Get_Index( state.get() )]
                        = std::thread( &Simulation_MMF_Start, state.get(), selected_solver, -1, -1, false, -1, -1 );
                }
                else if( selected_mode == GUI_Mode::EMA )
                {
                    int idx = System_Get_Index( state.get() );
                    if( threads_image[idx].joinable() )
                        threads_image[System_Get_Index( state.get() )].join();
                    threads_image[System_Get_Index( state.get() )]
                        = std::thread( &Simulation_EMA_Start, state.get(), -1, -1, false, -1, -1 );
                }
            }
        }

        if( ImGui::Button( ICON_FA_ARROW_LEFT, ImVec2( width, bar_height ) ) )
        {
        }

        static ImU32 image_number = (ImU32)1;
        ImGui::SetNextItemWidth( 40 );
        ImGui::InputScalar( "##imagenumber", ImGuiDataType_U32, &image_number, NULL, NULL, "%u" );

        if( ImGui::Button( ICON_FA_ARROW_RIGHT, ImVec2( width, bar_height ) ) )
        {
        }

        ImGui::PopStyleVar();

        ImGui::EndMainMenuBar();
    }
    ImGui::PopFont();
}

void show_energy_plot()
{
    auto & style = ImGui::GetStyle();

    static bool plot_image_energies        = true;
    static bool plot_interpolated_energies = true;

    static bool animate = true;
    // static float values[90]    = {};
    static std::vector<float> energies( 90, 0 );
    static int values_offset   = 0;
    static double refresh_time = 0.0;
    if( !animate || refresh_time == 0.0 )
        refresh_time = ImGui::GetTime();
    while( refresh_time < ImGui::GetTime() ) // Create dummy data at fixed 60 Hz rate for the demo
    {
        static float phase      = 0.0f;
        energies[values_offset] = cosf( phase );
        values_offset           = ( values_offset + 1 ) % energies.size();
        phase += 0.10f * values_offset;
        refresh_time += 1.0f / 60.0f;
    }

    ImGui::Begin( "Plots" );
    {
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
        if( ImGui::BeginTabBar( "plots_tab_bar", tab_bar_flags ) )
        {
            if( ImGui::BeginTabItem( "Energy" ) )
            {
                std::string overlay = fmt::format( "{:.3e}", energies[energies.size() - 1] );

                // ImGui::Text( "E" );
                // ImGui::SameLine();
                ImGui::PlotLines(
                    "", energies.data(), energies.size(), values_offset, overlay.c_str(), -1.0f, 1.0f,
                    ImVec2(
                        ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                        ImGui::GetWindowContentRegionMax().y - 110.f ) );

                ImGui::Checkbox( "Image energies", &plot_image_energies );
                ImGui::Checkbox( "Interpolated energies", &plot_interpolated_energies );
                ImGui::SameLine();
                static bool inputs_step = true;
                const ImU32 u32_one     = (ImU32)1;
                static ImU32 u32_v      = (ImU32)10;
                ImGui::PushItemWidth( 100 );
                ImGui::InputScalar(
                    "##energies", ImGuiDataType_U32, &u32_v, inputs_step ? &u32_one : NULL, NULL, "%u" );
                ImGui::PopItemWidth();

                ImGui::EndTabItem();
            }
            if( ImGui::BeginTabItem( "Convergence" ) )
            {
                std::string overlay = fmt::format( "{:.3e}", energies[energies.size() - 1] );

                // ImGui::Text( "F" );
                // ImGui::SameLine();
                ImGui::PlotLines(
                    "", energies.data(), energies.size(), values_offset, overlay.c_str(), -1.0f, 1.0f,
                    ImVec2(
                        ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                        ImGui::GetWindowContentRegionMax().y - 90.f ) );

                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void show_parameters( GUI_Mode & selected_mode )
{
    ImGui::Begin( "Parameters" );

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

void show_visualisation_settings( VFRendering::View & vfr_view, ImVec4 & background_colour )
{
    ImGui::Begin( "Visualisation settings" );

    ImGui::Text( "Background color" );
    if( ImGui::ColorEdit3( "##bgcolour", (float *)&background_colour ) )
    {
        vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { background_colour.x, background_colour.y, background_colour.z } );
    }

    ImGui::Separator();

    static vgm::Vec3 dir( 0, 0, -1 );
    ImGui::Text( "Light direction" );
    ImGui::Columns( 2, "lightdircolumns", false ); // 3-ways, no border
    if( ImGui::gizmo3D( "##dir", dir ) )
    {
        vfr_view.setOption<VFRendering::View::Option::LIGHT_POSITION>(
            { -1000 * dir.x, -1000 * dir.y, -1000 * dir.z } );
    }
    ImGui::NextColumn();
    ImGui::Text( fmt::format( "{:>6.3f}", dir.x ).c_str() );
    ImGui::Text( fmt::format( "{:>6.3f}", dir.y ).c_str() );
    ImGui::Text( fmt::format( "{:>6.3f}", dir.z ).c_str() );
    ImGui::Columns( 1 );
    ImGui::End();
}

void show_overlay_system( bool * p_open )
{
    static float energy = 0;
    static float m_x    = 0;
    static float m_y    = 0;
    static float m_z    = 0;

    static int noi           = 1;
    static int nos           = 1;
    static int n_basis_atoms = 1;
    static int n_a           = 1;
    static int n_b           = 1;
    static int n_c           = 1;

    const float DISTANCE = 50.0f;
    static int corner    = 0;

    ImGuiIO & io = ImGui::GetIO();

    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
    }
    ImGui::SetNextWindowBgAlpha( 0.35f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;
    if( ImGui::Begin( "System information overlay", p_open, window_flags ) )
    {
        ImGui::Text( fmt::format( "FPS: {:d}", int( io.Framerate ) ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "E      = {:.10f}", energy ).c_str() );
        ImGui::Text( fmt::format( "E dens = {:.10f}", energy / nos ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "M_x = {:.8f}", m_x ).c_str() );
        ImGui::Text( fmt::format( "M_y = {:.8f}", m_y ).c_str() );
        ImGui::Text( fmt::format( "M_z = {:.8f}", m_z ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "NOI: {}", noi ).c_str() );
        ImGui::Text( fmt::format( "NOS: {}", nos ).c_str() );
        ImGui::Text( fmt::format( "N basis atoms: {}", n_basis_atoms ).c_str() );
        ImGui::Text( fmt::format( "Cells: {}x{}x{}", n_a, n_b, n_c ).c_str() );

        ImGui::Separator();

        ImGui::Text( "Simple overlay\n"
                     "in the corner of the screen.\n"
                     "(right-click to change position)" );

        ImGui::Separator();

        if( ImGui::IsMousePosValid() )
            ImGui::Text( "Mouse Position: (%.1f,%.1f)", io.MousePos.x, io.MousePos.y );
        else
            ImGui::Text( "Mouse Position: <invalid>" );

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
            if( p_open && ImGui::MenuItem( "Close" ) )
                *p_open = false;
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void show_overlay_calculation( bool * p_open, GUI_Mode & selected_mode, int & selected_solver )
{
    static auto solvers_llg
        = std::map<int, std::pair<std::string, std::string>>{ { Solver_SIB, { "SIB", "Semi-implicit method B" } },
                                                              { Solver_Depondt, { "Depondt", "Depondt" } },
                                                              { Solver_Heun, { "Heun", "Heun" } },
                                                              { Solver_RungeKutta4,
                                                                { "RK4", "4th order Runge-Kutta" } } };

    static auto solvers_min = std::map<int, std::pair<std::string, std::string>>{
        { Solver_VP, { "VP", "Velocity Projection" } },
        { Solver_VP_OSO, { "VP (OSO)", "Velocity Projection (OSO)" } },
        { Solver_SIB, { "SIB", "Semi-implicit method B" } },
        { Solver_Depondt, { "Depondt", "Depondt" } },
        { Solver_Heun, { "Heun", "Heun" } },
        { Solver_RungeKutta4, { "RK4", "4th order Runge-Kutta" } },
        { Solver_LBFGS_OSO, { "LBFGS (OSO)", "LBFGS (OSO)" } },
        { Solver_LBFGS_Atlas, { "LBFGS (Atlas)", "LBFGS (Atlas)" } }
    };

    static float simulated_time = 0;
    static float wall_time      = 0;
    static int iteration        = 0;
    static float ips            = 0;

    int hours        = wall_time / ( 60 * 60 * 1000 );
    int minutes      = ( wall_time - 60 * 60 * 1000 * hours ) / ( 60 * 1000 );
    int seconds      = ( wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes ) / 1000;
    int milliseconds = wall_time - 60 * 60 * 1000 * hours - 60 * 1000 * minutes - 1000 * seconds;

    static float force_max = 0;

    const float DISTANCE = 50.0f;
    static int corner    = 1;

    ImGuiIO & io = ImGui::GetIO();
    if( corner != -1 )
    {
        ImVec2 window_pos = ImVec2(
            ( corner & 1 ) ? io.DisplaySize.x - DISTANCE : DISTANCE,
            ( corner & 2 ) ? io.DisplaySize.y - DISTANCE : DISTANCE );
        ImVec2 window_pos_pivot = ImVec2( ( corner & 1 ) ? 1.0f : 0.0f, ( corner & 2 ) ? 1.0f : 0.0f );
        ImGui::SetNextWindowPos( window_pos, ImGuiCond_Always, window_pos_pivot );
    }
    ImGui::SetNextWindowBgAlpha( 0.35f ); // Transparent background
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                    | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                    | ImGuiWindowFlags_NoNav;
    if( corner != -1 )
        window_flags |= ImGuiWindowFlags_NoMove;
    if( ImGui::Begin( "Calculation information overlay", p_open, window_flags ) )
    {
        if( selected_mode == GUI_Mode::Minimizer || selected_mode == GUI_Mode::LLG || selected_mode == GUI_Mode::GNEB
            || selected_mode == GUI_Mode::MMF )
        {
            auto & solvers = solvers_min;
            if( selected_mode == GUI_Mode::LLG )
                solvers = solvers_llg;

            if( ImGui::Button( fmt::format( "Solver: {}", solvers[selected_solver].first ).c_str() ) )
                ImGui::OpenPopup( "solver_popup" );
            if( ImGui::BeginPopup( "solver_popup" ) )
            {
                // ImGui::Text( "Aquarium" );
                for( auto solver : solvers )
                    if( ImGui::Selectable( solver.second.first.c_str() ) )
                        selected_solver = solver.first;
                ImGui::EndPopup();
            }
            ImGui::Separator();
        }
        if( selected_mode == GUI_Mode::LLG )
        {
            ImGui::Text( fmt::format( "t = {} ps", simulated_time ).c_str() );
        }

        ImGui::Text( fmt::format( "{:0>2d}:{:0>2d}:{:0>2d}.{:0>3d}", hours, minutes, seconds, milliseconds ).c_str() );
        ImGui::Text( fmt::format( "Iteration: {}", iteration ).c_str() );
        ImGui::Text( fmt::format( "IPS: {:.2f}", ips ).c_str() );

        ImGui::Separator();

        ImGui::Text( fmt::format( "F_max = {:.5e}", force_max ).c_str() );
        if( selected_mode == GUI_Mode::GNEB )
        {
            ImGui::Text( fmt::format( "F_current = {:.5e}", simulated_time ).c_str() );
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
            if( p_open && ImGui::MenuItem( "Close" ) )
                *p_open = false;
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void show_keybindings( bool & show_keybindings )
{
    if( !show_keybindings )
        return;

    ImGui::Begin( "Keybindings" );

    ImGui::Text( "UI controls" );
    ImGui::BulletText( "F1: Show this" );
    ImGui::BulletText( "F2: Toggle settings" );
    ImGui::BulletText( "F3: Toggle plots" );
    ImGui::BulletText( "F4: Toggle debug" );
    ImGui::BulletText( "F5: Toggle \"Dragging\" mode" );
    ImGui::BulletText( "F6: Toggle \"Defects\" mode" );
    ImGui::BulletText( "F7: Toggle \"Pinning\" mode" );
    ImGui::Text( "" );
    ImGui::BulletText( "F10  and Ctrl+F:      Toggle large visualisation" );
    ImGui::BulletText( "F11 and Ctrl+Shift+F: Toggle fullscreen window" );
    ImGui::BulletText( "F12 and Home:         Screenshot of visualisation region" );
    ImGui::BulletText( "Ctrl+Shift+V:         Toggle OpenGL visualisation" );
    ImGui::BulletText( "i:                    Toggle large visualisation" );
    ImGui::BulletText( "Escape:               Try to return focus to main UI (does not always work)" );
    ImGui::Text( "" );
    ImGui::Text( "Camera controls" );
    ImGui::BulletText( "Left mouse:   Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Right mouse:  Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "Scroll mouse: Zoom in on focus point (<b>shift</b> to go slow)" );
    ImGui::BulletText( "WASD:         Rotate the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "TFGH:         Move the camera around (<b>shift</b> to go slow)" );
    ImGui::BulletText( "X,Y,Z:        Set the camera in X, Y or Z direction (<b>shift</b> to invert)" );
    ImGui::Text( "" );
    ImGui::Text( "Control Simulations" );
    ImGui::BulletText( "Space:        Start/stop calculation" );
    ImGui::BulletText( "Ctrl+M:       Cycle method" );
    ImGui::BulletText( "Ctrl+S:       Cycle solver" );
    ImGui::Text( "" );
    ImGui::Text( "Manipulate the current images" );
    ImGui::BulletText( "Ctrl+R:       Random configuration" );
    ImGui::BulletText( "Ctrl+N:       Add tempered noise" );
    ImGui::BulletText( "Enter:        Insert last used configuration" );
    ImGui::Text( "" );
    ImGui::Text( "Visualisation" );
    ImGui::BulletText( "+/-:          Use more/fewer data points of the vector field" );
    ImGui::BulletText( "1:            Regular Visualisation Mode" );
    ImGui::BulletText( "2:            Isosurface Visualisation Mode" );
    ImGui::BulletText( "3-5:          Slab (X,Y,Z) Visualisation Mode" );
    ImGui::BulletText( "/:            Cycle Visualisation Mode" );
    ImGui::BulletText( ", and .:      Move Slab (<b>shift</b> to go faster)" );
    ImGui::Text( "" );
    ImGui::Text( "Manipulate the chain of images" );
    ImGui::BulletText( "Arrows:          Switch between images and chains" );
    ImGui::BulletText( "Ctrl+X:          Cut   image" );
    ImGui::BulletText( "Ctrl+C:          Copy  image" );
    ImGui::BulletText( "Ctrl+V:          Paste image at current index" );
    ImGui::BulletText( "Ctrl+Left/Right: Insert left/right of current index<" );
    ImGui::BulletText( "Del:             Delete image" );
    ImGui::Text( "" );
    ImGui::TextWrapped( "Note that some of the keybindings may only work correctly on US keyboard layout.\n"
                        "\n"
                        "For more information see the documentation at spirit-docs.readthedocs.io" );
    ImGui::Text( "" );

    if( ImGui::Button( "Close" ) )
        show_keybindings = false;
    ImGui::End();
}

void show_about( bool & show_about )
{
    if( !show_about )
        return;

    ImGui::Begin( "About" );

    ImGui::Text( fmt::format( "Library version {}", Spirit_Version_Full() ).c_str() );
    ImGui::Text( "" );
    ImGui::TextWrapped( "The <b>Spirit</b> GUI application incorporates intuitive visualisation,"
                        "powerful <b>Spin Dynamics</b> and <b>Nudged Elastic Band</b> tools"
                        "into a cross-platform user interface." );
    ImGui::Text( "" );
    ImGui::Text( "Main developers:" );
    ImGui::BulletText(
        "Moritz Sallermann (<a href=\"mailto:m.sallermann@fz-juelich.de\">m.sallermann@fz-juelich.de</a>)" );
    ImGui::BulletText( "Gideon Mueller (<a href=\"mailto:g.mueller@fz-juelich.de\">g.mueller@fz-juelich.de</a>)" );
    ImGui::TextWrapped(
        "at the Institute for Advanced Simulation 1 of the Forschungszentrum Juelich.\n"
        "For more information about us, visit <a href=\"http://juspin.de\">juSpin.de</a>"
        " or see the <a href=\"http://www.fz-juelich.de/pgi/pgi-1/DE/Home/home_node.html\">IAS-1 Website</a>." );
    ImGui::Text( "" );
    ImGui::TextWrapped( "The sources are hosted at <a href=\"https://spirit-code.github.io\">spirit-code.github.io</a>"
                        " and the documentation can be found at <a "
                        "href=\"https://spirit-docs.readthedocs.io\">spirit-docs.readthedocs.io</a>." );
    ImGui::Text( "" );

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