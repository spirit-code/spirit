#pragma once
#ifndef SPIRIT_IMGUI_UI_SHARED_STATE_HPP
#define SPIRIT_IMGUI_UI_SHARED_STATE_HPP

#include <enums.hpp>

#include <Spirit/Simulation.h>

#include <array>
#include <list>
#include <string>

namespace ui
{

struct UiSharedState
{
    struct Notification
    {
        std::string message = "";
        float timer         = 0;
        float timeout       = 3;
    };

    enum class InteractionMode
    {
        REGULAR,
        DRAG,
        DEFECT,
        PINNING
    };

    void notify( const std::string & notification, float timeout = 3 )
    {
        this->notifications.push_back( Notification{ notification, 0, timeout > 1 ? timeout : 1 } );
    }

    // Simulation
    int selected_solver_min = Solver_VP_OSO;
    int selected_solver_llg = Solver_Depondt;
    GUI_Mode selected_mode  = GUI_Mode::Minimizer;

    // Interaction
    InteractionMode interaction_mode = InteractionMode::REGULAR;

    // Window and visualisation state
    bool dark_mode = true;
    std::array<float, 3> background_dark{ 0.4f, 0.4f, 0.4f };
    std::array<float, 3> background_light{ 0.9f, 0.9f, 0.9f };
    std::array<float, 3> light_direction{ 0, 0, -1 };

    // Camera in regular interaction mode
    bool camera_is_orthographic  = false;
    float camera_perspective_fov = 60;
    std::array<float, 3> regular_interaction_camera_pos;
    std::array<float, 3> regular_interaction_center_pos;
    std::array<float, 3> regular_interaction_sys_center;
    std::array<float, 3> regular_interaction_camera_up;

    // Other
    int n_screenshots = 0;
    std::list<Notification> notifications;
};

} // namespace ui

#endif