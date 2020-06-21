#include <fonts.hpp>

#include <imgui_impl/font_cousine_regular.h>
#include <imgui_impl/fontawesome_regular.h>

#include <imgui/imgui.h>

namespace fonts
{

ImFont * imgui_default( float size_px )
{
    ImFontConfig default_font_cfg = ImFontConfig();
    default_font_cfg.SizePixels   = size_px;

    ImGuiIO & io = ImGui::GetIO();
    return io.Fonts->AddFontDefault( &default_font_cfg );
}

ImFont * cousine_regular( float size_px )
{
    static const ImWchar ranges[] = { 0x0020, 0x00FF, 0x0100, 0x017F, 0 };

    ImGuiIO & io = ImGui::GetIO();
    return io.Fonts->AddFontFromMemoryCompressedTTF(
        cousine_regular_compressed_data, cousine_regular_compressed_size, size_px, NULL, ranges );
}

ImFont * fontawesome_icons( float size_px )
{
    static const ImWchar ranges[] = { 0xf000, 0xf976, 0 };

    ImFontConfig config;
    ImGuiIO & io      = ImGui::GetIO();
    config.PixelSnapH = true;
    return io.Fonts->AddFontFromMemoryCompressedTTF(
        fontawesome_regular_compressed_data, fontawesome_regular_compressed_size, size_px, &config, ranges );
}

ImFont * font_combined( float size_px )
{
    static const ImWchar cousine_ranges[]     = { 0x0020, 0x00FF, 0x0100, 0x017F, 0 };
    static const ImWchar fontawesome_ranges[] = { 0xf000, 0xf976, 0 };

    ImFontConfig config;

    ImGuiIO & io = ImGui::GetIO();
    io.Fonts->AddFontFromMemoryCompressedTTF(
        cousine_regular_compressed_data, cousine_regular_compressed_size, size_px, &config, cousine_ranges );
    config.MergeMode  = true;
    config.PixelSnapH = true;
    return io.Fonts->AddFontFromMemoryCompressedTTF(
        fontawesome_regular_compressed_data, fontawesome_regular_compressed_size, size_px, &config,
        fontawesome_ranges );
}

} // namespace fonts