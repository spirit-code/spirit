#pragma once
#ifndef SPIRIT_IMGUI_WIDGETS_HPP
#define SPIRIT_IMGUI_WIDGETS_HPP

#include <enums.hpp>

#include <GLFW/glfw3.h>

#include <VFRendering/View.hxx>

#include <thread>
#include <vector>

struct State;

namespace widgets
{

void show_menu_bar(
    GLFWwindow * window, ImFont * font, bool & dark_mode, ImVec4 & background_colour, GUI_Mode & selected_mode,
    int & selected_solver, VFRendering::View & vfr_view, bool & show_keybindings, bool & show_overlays,
    bool & show_about, std::shared_ptr<State> state, std::vector<std::thread> & threads_image,
    std::thread & thread_chain );
void help_marker( const char * description );

void show_parameters( GUI_Mode & selected_mode, bool & show );
void show_visualisation_settings( VFRendering::View & vfr_view, ImVec4 & background_colour );

void show_energy_plot();

void show_overlay_system( bool & show );
void show_overlay_calculation( bool & show, GUI_Mode & selected_mode, int & selected_solver );

void show_keybindings( bool & show_keybindings );
void show_about( bool & show_about );

} // namespace widgets

#endif