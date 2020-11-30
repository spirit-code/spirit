#pragma once
#ifndef SPIRIT_IMGUI_UI_CONFIG_FILE_HPP
#define SPIRIT_IMGUI_UI_CONFIG_FILE_HPP

#include <ui_shared_state.hpp>

#include <array>
#include <string>

namespace ui
{

struct UiConfigFile
{
    UiConfigFile( UiSharedState & ui_shared_state );
    ~UiConfigFile();

    void to_json() const;
    void from_json();

    const std::string settings_filename = "spirit_settings.json";

    UiSharedState & ui_shared_state;

    // Window state
    bool maximized  = false;
    bool fullscreen = false;
    std::array<int, 2> window_position{ 100, 100 };
    std::array<int, 2> window_size{ 1280, 720 };
};

} // namespace ui

#endif