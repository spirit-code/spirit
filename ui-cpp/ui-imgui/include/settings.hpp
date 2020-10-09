#pragma once
#ifndef SPIRIT_IMGUI_SETTINGS_HPP
#define SPIRIT_IMGUI_SETTINGS_HPP

#include <enums.hpp>

#include <Spirit/Simulation.h>

namespace ui
{

struct Settings
{
    // Simulation
    int selected_solver_min = Solver_VP_OSO;
    int selected_solver_llg = Solver_Depondt;
    GUI_Mode selected_mode  = GUI_Mode::Minimizer;

    // Window state
    bool dark_mode  = true;
    bool maximized  = false;
    bool fullscreen = false;

    // Interaction
    bool dragging_mode = false;
};

} // namespace ui

#endif