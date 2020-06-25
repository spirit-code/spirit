#include <fonts.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <imgui/imgui.h>

#include <fmt/format.h>

#include <nfd.h>

#include <map>
#include <string>

static auto modes = std::map<GUI_Mode, std::pair<std::string, std::string>>{
    { GUI_Mode::Minimizer, { "Minimizer", "(1) energy minimisation" } },
    { GUI_Mode::MC, { "Monte Carlo", "(2) Monte Carlo Stochastical sampling" } },
    { GUI_Mode::LLG, { "LLG", "(3) Landau-Lifshitz-Gilbert dynamics" } },
    { GUI_Mode::GNEB, { "GNEB", "(4) geodesic nudged elastic band calculation" } },
    { GUI_Mode::EMA, { "Eigenmodes", "(5) eigenmode calculation and visualisation" } }
};

namespace widgets
{

void show_menu_bar(
    GLFWwindow * window, ImFont * font, bool & dark_mode, ImVec4 & background_colour, GUI_Mode & selected_mode,
    VFRendering::View & vfr_view )
{
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 7.f, 7.f ) );
    ImGui::PushFont( font );

    if( ImGui::BeginMainMenuBar() )
    {
        // mainmenu_height = ImGui::GetWindowSize().y;
        ImGui::PopStyleVar();

        // ImGui::SameLine();
        if( ImGui::BeginMenu( "File" ) )
        {
            if( ImGui::MenuItem( "New" ) )
            {
            }
            if( ImGui::MenuItem( "Open", "ctrl+o" ) )
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
                else if( result == NFD_CANCEL )
                {
                    fmt::print( "User pressed cancel.\n" );
                }
                else
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save", "ctrl+s" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "ovf;txt;csv", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
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
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Edit" ) )
        {
            if( ImGui::MenuItem( "Undo", "ctrl+z" ) )
            {
            }
            if( ImGui::MenuItem( "Redo", "ctrl+y", false, false ) )
            {
            } // Disabled item
            ImGui::Separator();
            if( ImGui::MenuItem( "Cut", "ctrl+x" ) )
            {
            }
            if( ImGui::MenuItem( "Copy", "ctrl+c" ) )
            {
            }
            if( ImGui::MenuItem( "Paste", "ctrl+v" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "View" ) )
        {
            if( ImGui::MenuItem( "Something" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Fullscreen", "ctrl+shift+f" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Help" ) )
        {
            if( ImGui::MenuItem( "Keybindings", "F1" ) )
            {
            }
            if( ImGui::MenuItem( "About" ) )
            {
            }
            ImGui::EndMenu();
        }

        auto io            = ImGui::GetIO();
        auto & style       = ImGui::GetStyle();
        float font_size_px = font->FontSize;
        float right_edge   = ImGui::GetWindowContentRegionMax().x;
        float bar_height   = ImGui::GetWindowContentRegionMax().y + 2 * style.FramePadding.y;
        float width;

        ImGui::PushStyleVar( ImGuiStyleVar_SelectableTextAlign, ImVec2( .5f, .5f ) );

        width = 2.5f * font_size_px;
        ImGui::SameLine( right_edge - width );
        if( dark_mode )
        {
            if( ImGui::Button( ICON_FA_SUN, ImVec2( width, bar_height ) ) )
            {
                ImGui::StyleColorsLight();
                background_colour = ImVec4( 0.7f, 0.7f, 0.7f, 1.f );
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

        for( int n = modes.size(); n > 0; n-- )
        {
            auto mode         = GUI_Mode( n );
            std::string label = modes[mode].first;
            width             = label.length() * font_size_px * 0.6;
            ImGui::SameLine( right_edge - width );
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

        ImGui::PopStyleVar();

        ImGui::EndMainMenuBar();
    }
    ImGui::PopFont();
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

void show_overlay( bool * p_open )
{
    const float DISTANCE = 50.0f;
    static int corner    = 0;
    ImGuiIO & io         = ImGui::GetIO();
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
    if( ImGui::Begin( "Example: Simple overlay", p_open, window_flags ) )
    {
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

} // namespace widgets