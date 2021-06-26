#pragma once
#ifndef SPIRIT_IMGUI_UI_CONFIG_FILE_HPP
#define SPIRIT_IMGUI_UI_CONFIG_FILE_HPP

#include <rendering_layer.hpp>
#include <ui_shared_state.hpp>

#include <array>
#include <string>

namespace ui
{

struct UiConfigFile
{
    UiConfigFile( UiSharedState & ui_shared_state, RenderingLayer & rendering_layer );
    ~UiConfigFile();

    void to_json() const;
    void from_json();

    const std::string settings_filename = "spirit_settings.json";

    UiSharedState & ui_shared_state;
    RenderingLayer & rendering_layer;

    // Which windows to show
    bool show_configurations_widget = true;
    bool show_parameters_widget     = false;
    bool show_visualisation_widget  = false;
    bool show_hamiltonian_widget    = false;
    bool show_geometry_widget       = false;
    bool show_plots                 = true;
    bool show_settings              = false;

    // Overlays
    bool show_overlays        = true;
    int overlay_system_corner = 0;
    std::array<float, 2> overlay_system_position;
    int overlay_calculation_corner = 1;
    std::array<float, 2> overlay_calculation_position;

    // Interaction
    float interaction_radius = 80;

    // Main window
    bool window_maximized  = false;
    bool window_fullscreen = false;
    std::array<int, 2> window_position{ 100, 100 };
    std::array<int, 2> window_size{ 1280, 720 };
};

} // namespace ui

#endif