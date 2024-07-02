#pragma once
#ifndef SPIRIT_IMGUI_UI_SHARED_STATE_HPP
#define SPIRIT_IMGUI_UI_SHARED_STATE_HPP

#include <enums.hpp>

#include <Spirit/Simulation.h>

#include <algorithm>
#include <array>
#include <deque>
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
        this->notifications.push_back( { notification, 0, timeout > 1 ? timeout : 1 } );
    }

    void expire_notifications( const float deltaTime = 0 )
    {
        // advance internal clock of each notification
        std::for_each(
            notifications.begin(), notifications.end(),
            [deltaTime]( Notification & notification ) { notification.timer += deltaTime; } );
        // erase from the front to minimize move operations (theoreticaly these should already be in order)
        notifications.erase(
            notifications.begin(),
            std::remove_if(
                notifications.rbegin(), notifications.rend(),
                []( const Notification & notification ) { return notification.timer > notification.timeout; } )
                .base() );
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
    bool camera_is_orthographic = false;
    int camera_perspective_fov  = 45;
    std::array<float, 3> regular_interaction_camera_pos;
    std::array<float, 3> regular_interaction_center_pos;
    std::array<float, 3> regular_interaction_sys_center;
    std::array<float, 3> regular_interaction_camera_up;

    // Parameters widget
    bool min_apply_to_all = true;
    bool mc_apply_to_all  = true;
    bool llg_apply_to_all = true;
    bool mmf_apply_to_all = true;
    bool ema_apply_to_all = true;

    // Configurations widget
    struct Configurations
    {
        std::string last_used = "";

        scalar pos[3]{ 0, 0, 0 };
        scalar border_rect[3]{ -1, -1, -1 };
        scalar border_cyl = -1;
        scalar border_sph = -1;
        bool inverted     = false;

        scalar noise_temperature = 0;
    };
    Configurations configurations{};

    // Other
    int n_screenshots       = 0;
    int n_screenshots_chain = 0;
    std::deque<Notification> notifications;
};

} // namespace ui

#endif