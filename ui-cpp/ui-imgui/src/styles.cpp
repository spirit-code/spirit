#include <styles.hpp>

namespace styles
{

void apply_charcoal( ImGuiStyle * dst )
{
    ImGuiStyle & style = dst ? *dst : ImGui::GetStyle();

    style.Alpha             = 1.0f;
    style.ChildRounding     = 4.0f;
    style.FrameBorderSize   = 1.0f;
    style.FrameRounding     = 2.0f;
    style.GrabMinSize       = 7.0f;
    style.PopupRounding     = 2.0f;
    style.ScrollbarRounding = 12.0f;
    style.ScrollbarSize     = 13.0f;
    style.WindowRounding    = 4.0f;

    ImVec4 * colors = style.Colors;

    colors[ImGuiCol_Text]                 = ImVec4( 0.900f, 0.900f, 0.900f, 1.000f );
    colors[ImGuiCol_TextDisabled]         = ImVec4( 0.500f, 0.500f, 0.500f, 1.000f );
    colors[ImGuiCol_WindowBg]             = ImVec4( 0.180f, 0.180f, 0.180f, 1.000f );
    colors[ImGuiCol_ChildBg]              = ImVec4( 0.280f, 0.280f, 0.280f, 0.000f );
    colors[ImGuiCol_PopupBg]              = ImVec4( 0.313f, 0.313f, 0.313f, 1.000f );
    colors[ImGuiCol_Border]               = ImVec4( 0.266f, 0.266f, 0.266f, 0.300f );
    colors[ImGuiCol_BorderShadow]         = ImVec4( 0.000f, 0.000f, 0.000f, 0.000f );
    colors[ImGuiCol_FrameBg]              = ImVec4( 0.160f, 0.160f, 0.160f, 1.000f );
    colors[ImGuiCol_FrameBgHovered]       = ImVec4( 0.200f, 0.200f, 0.200f, 1.000f );
    colors[ImGuiCol_FrameBgActive]        = ImVec4( 0.280f, 0.280f, 0.280f, 1.000f );
    colors[ImGuiCol_TitleBg]              = ImVec4( 0.148f, 0.148f, 0.148f, 1.000f );
    colors[ImGuiCol_TitleBgActive]        = ImVec4( 0.148f, 0.148f, 0.148f, 1.000f );
    colors[ImGuiCol_TitleBgCollapsed]     = ImVec4( 0.148f, 0.148f, 0.148f, 1.000f );
    colors[ImGuiCol_MenuBarBg]            = ImVec4( 0.195f, 0.195f, 0.195f, 1.000f );
    colors[ImGuiCol_ScrollbarBg]          = ImVec4( 0.160f, 0.160f, 0.160f, 1.000f );
    colors[ImGuiCol_ScrollbarGrab]        = ImVec4( 0.277f, 0.277f, 0.277f, 1.000f );
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4( 0.300f, 0.300f, 0.300f, 1.000f );
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_CheckMark]            = ImVec4( 1.000f, 1.000f, 1.000f, 1.000f );
    colors[ImGuiCol_SliderGrab]           = ImVec4( 0.391f, 0.391f, 0.391f, 1.000f );
    colors[ImGuiCol_SliderGrabActive]     = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_Button]               = ImVec4( 1.000f, 1.000f, 1.000f, 0.165f );
    colors[ImGuiCol_ButtonHovered]        = ImVec4( 1.000f, 1.000f, 1.000f, 0.205f );
    colors[ImGuiCol_ButtonActive]         = ImVec4( 1.000f, 1.000f, 1.000f, 0.391f );
    colors[ImGuiCol_Header]               = ImVec4( 0.313f, 0.313f, 0.313f, 1.000f );
    colors[ImGuiCol_HeaderHovered]        = ImVec4( 0.469f, 0.469f, 0.469f, 1.000f );
    colors[ImGuiCol_HeaderActive]         = ImVec4( 0.469f, 0.469f, 0.469f, 1.000f );
    colors[ImGuiCol_Separator]            = ImVec4( 1.000f, 1.000f, 1.000f, 0.277f );
    colors[ImGuiCol_SeparatorHovered]     = ImVec4( 1.000f, 1.000f, 1.000f, 0.469f );
    colors[ImGuiCol_SeparatorActive]      = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_ResizeGrip]           = ImVec4( 1.000f, 1.000f, 1.000f, 0.250f );
    colors[ImGuiCol_ResizeGripHovered]    = ImVec4( 1.000f, 1.000f, 1.000f, 0.670f );
    colors[ImGuiCol_ResizeGripActive]     = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_Tab]                  = colors[ImGuiCol_Header];
    colors[ImGuiCol_TabHovered]           = colors[ImGuiCol_HeaderHovered];
    colors[ImGuiCol_TabActive]            = colors[ImGuiCol_HeaderActive];
    colors[ImGuiCol_TabUnfocused]         = colors[ImGuiCol_Tab];
    colors[ImGuiCol_TabUnfocusedActive]   = colors[ImGuiCol_TabActive];
    colors[ImGuiCol_PlotLines]            = ImVec4( 0.469f, 0.469f, 0.469f, 1.000f );
    colors[ImGuiCol_PlotLinesHovered]     = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_PlotHistogram]        = ImVec4( 0.586f, 0.586f, 0.586f, 1.000f );
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_TextSelectedBg]       = ImVec4( 1.000f, 1.000f, 1.000f, 0.156f );
    colors[ImGuiCol_DragDropTarget]       = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
    colors[ImGuiCol_NavHighlight]         = ImVec4( 1.000f, 0.391f, 0.000f, 1.000f );
}

} // namespace styles