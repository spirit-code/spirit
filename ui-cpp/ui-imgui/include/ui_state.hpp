#pragma once
#ifndef SPIRIT_IMGUI_UI_STATE_HPP
#define SPIRIT_IMGUI_UI_STATE_HPP

#include <enums.hpp>

#include <Spirit/Simulation.h>

#include <array>
#include <list>
#include <string>

namespace ui
{

struct UiState
{
    struct Notification
    {
        std::string message = "";
        float timer         = 0;
        float timeout       = 3;
    };

    void notify( const std::string & notification, float timeout = 3 )
    {
        this->notifications.push_back( Notification{ notification, 0, timeout > 1 ? timeout : 1 } );
    }

    UiState();

    void to_json() const;
    void from_json();

    const std::string settings_filename = "settings.json";

    // Simulation
    int selected_solver_min = Solver_VP_OSO;
    int selected_solver_llg = Solver_Depondt;
    GUI_Mode selected_mode  = GUI_Mode::Minimizer;

    // Window state
    bool dark_mode = true;
    std::array<float, 3> background_dark{ 0.9f, 0.0f, 0.2f };
    std::array<float, 3> background_light{ 0.9f, 0.0f, 0.2f };
    std::array<float, 3> light_direction{ 0, 0, -1 };
    bool maximized  = false;
    bool fullscreen = false;
    std::array<int, 2> pos{ 100, 100 };
    std::array<int, 2> size{ 1280, 720 };

    // Interaction
    bool dragging_mode = false;

    // Other
    int n_screenshots = 0;
    std::list<Notification> notifications;
};

} // namespace ui

#endif