#include <styles.hpp>

namespace styles
{

template<typename T>
static inline T lerp( T a, T b, float t )
{
    return static_cast<T>( a + ( b - a ) * t );
}

static inline ImVec4 lerp( const ImVec4 & a, const ImVec4 & b, float t )
{
    return ImVec4( lerp( a.x, b.x, t ), lerp( a.y, b.y, t ), lerp( a.z, b.z, t ), lerp( a.w, b.w, t ) );
}

void apply_light( ImGuiStyle * dst )
{
    ImGuiStyle & style = dst ? *dst : ImGui::GetStyle();

    style.Alpha             = 1.0f;
    style.ChildRounding     = 6.0f;
    style.FrameBorderSize   = 1.0f;
    style.FrameRounding     = 4.0f;
    style.GrabMinSize       = 7.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 12.0f;
    style.ScrollbarSize     = 13.0f;
    style.WindowRounding    = 6.0f;
    style.WindowBorderSize  = 2.0f;

    ImVec4 * colors = style.Colors;

    colors[ImGuiCol_Text]                 = ImVec4( 0.18f, 0.18f, 0.18f, 1.00f );
    colors[ImGuiCol_TextDisabled]         = ImVec4( 0.60f, 0.60f, 0.60f, 1.00f );
    colors[ImGuiCol_WindowBg]             = ImVec4( 0.94f, 0.94f, 0.94f, 1.00f );
    colors[ImGuiCol_ChildBg]              = ImVec4( 0.00f, 0.00f, 0.00f, 0.00f );
    colors[ImGuiCol_PopupBg]              = ImVec4( 1.00f, 1.00f, 1.00f, 0.98f );
    colors[ImGuiCol_DockingEmptyBg]       = ImVec4( 1.00f, 1.00f, 1.00f, 0.00f );
    colors[ImGuiCol_Border]               = ImVec4( 0.77f, 0.77f, 0.77f, 0.20f );
    colors[ImGuiCol_BorderShadow]         = ImVec4( 0.00f, 0.00f, 0.00f, 0.00f );
    colors[ImGuiCol_FrameBg]              = ImVec4( 1.00f, 1.00f, 1.00f, 1.00f );
    colors[ImGuiCol_FrameBgHovered]       = ImVec4( 0.26f, 0.59f, 0.98f, 0.40f );
    colors[ImGuiCol_FrameBgActive]        = ImVec4( 0.26f, 0.59f, 0.98f, 0.67f );
    colors[ImGuiCol_TitleBg]              = ImVec4( 0.96f, 0.96f, 0.96f, 1.00f );
    colors[ImGuiCol_TitleBgActive]        = ImVec4( 0.82f, 0.82f, 0.82f, 1.00f );
    colors[ImGuiCol_TitleBgCollapsed]     = ImVec4( 1.00f, 1.00f, 1.00f, 0.51f );
    colors[ImGuiCol_MenuBarBg]            = ImVec4( 0.86f, 0.86f, 0.86f, 1.00f );
    colors[ImGuiCol_ScrollbarBg]          = ImVec4( 0.98f, 0.98f, 0.98f, 0.53f );
    colors[ImGuiCol_ScrollbarGrab]        = ImVec4( 0.69f, 0.69f, 0.69f, 0.80f );
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4( 0.49f, 0.49f, 0.49f, 0.80f );
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4( 0.49f, 0.49f, 0.49f, 1.00f );
    colors[ImGuiCol_CheckMark]            = ImVec4( 0.26f, 0.59f, 0.98f, 1.00f );
    colors[ImGuiCol_SliderGrab]           = ImVec4( 0.26f, 0.59f, 0.98f, 0.78f );
    colors[ImGuiCol_SliderGrabActive]     = ImVec4( 0.46f, 0.54f, 0.80f, 0.60f );
    colors[ImGuiCol_Button]               = ImVec4( 0.26f, 0.59f, 0.98f, 0.40f );
    colors[ImGuiCol_ButtonHovered]        = ImVec4( 0.26f, 0.59f, 0.98f, 1.00f );
    colors[ImGuiCol_ButtonActive]         = ImVec4( 0.06f, 0.53f, 0.98f, 1.00f );
    colors[ImGuiCol_Header]               = ImVec4( 0.26f, 0.59f, 0.98f, 0.31f );
    colors[ImGuiCol_HeaderHovered]        = ImVec4( 0.26f, 0.59f, 0.98f, 0.80f );
    colors[ImGuiCol_HeaderActive]         = ImVec4( 0.26f, 0.59f, 0.98f, 1.00f );
    colors[ImGuiCol_Separator]            = ImVec4( 0.39f, 0.39f, 0.39f, 0.52f );
    colors[ImGuiCol_SeparatorHovered]     = ImVec4( 0.62f, 0.62f, 0.62f, 0.58f );
    colors[ImGuiCol_SeparatorActive]      = ImVec4( 0.14f, 0.44f, 0.80f, 1.00f );
    colors[ImGuiCol_ResizeGrip]           = ImVec4( 0.80f, 0.80f, 0.80f, 0.56f );
    colors[ImGuiCol_ResizeGripHovered]    = ImVec4( 0.26f, 0.59f, 0.98f, 0.67f );
    colors[ImGuiCol_ResizeGripActive]     = ImVec4( 0.26f, 0.59f, 0.98f, 0.95f );
    colors[ImGuiCol_Tab]                  = lerp( colors[ImGuiCol_Header], colors[ImGuiCol_TitleBgActive], 0.90f );
    colors[ImGuiCol_TabHovered]           = colors[ImGuiCol_HeaderHovered];
    colors[ImGuiCol_TabActive]          = lerp( colors[ImGuiCol_HeaderActive], colors[ImGuiCol_TitleBgActive], 0.60f );
    colors[ImGuiCol_TabUnfocused]       = lerp( colors[ImGuiCol_Tab], colors[ImGuiCol_TitleBg], 0.80f );
    colors[ImGuiCol_TabUnfocusedActive] = lerp( colors[ImGuiCol_TabActive], colors[ImGuiCol_TitleBg], 0.40f );
    colors[ImGuiCol_PlotLines]          = ImVec4( 0.39f, 0.39f, 0.39f, 1.00f );
    colors[ImGuiCol_PlotLinesHovered]   = ImVec4( 1.00f, 0.43f, 0.35f, 1.00f );
    colors[ImGuiCol_PlotHistogram]      = ImVec4( 0.90f, 0.70f, 0.00f, 1.00f );
    colors[ImGuiCol_PlotHistogramHovered]  = ImVec4( 1.00f, 0.45f, 0.00f, 1.00f );
    colors[ImGuiCol_TextSelectedBg]        = ImVec4( 0.26f, 0.59f, 0.98f, 0.35f );
    colors[ImGuiCol_DragDropTarget]        = ImVec4( 0.26f, 0.59f, 0.98f, 0.95f );
    colors[ImGuiCol_NavHighlight]          = colors[ImGuiCol_HeaderHovered];
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4( 0.70f, 0.70f, 0.70f, 0.70f );
    colors[ImGuiCol_NavWindowingDimBg]     = ImVec4( 0.20f, 0.20f, 0.20f, 0.20f );
    colors[ImGuiCol_ModalWindowDimBg]      = ImVec4( 0.20f, 0.20f, 0.20f, 0.35f );
}

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
    colors[ImGuiCol_DockingEmptyBg]       = ImVec4( 1.000f, 1.000f, 1.000f, 0.000f );
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