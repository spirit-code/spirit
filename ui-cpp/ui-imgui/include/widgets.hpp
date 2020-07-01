#pragma once
#ifndef SPIRIT_IMGUI_WIDGETS_HPP
#define SPIRIT_IMGUI_WIDGETS_HPP

#include <enums.hpp>

#include <GLFW/glfw3.h>

#include <VFRendering/View.hxx>

#include <thread>
#include <vector>

namespace widgets
{

void show_menu_bar(
    GLFWwindow * window, ImFont * font, bool & dark_mode, ImVec4 & background_colour, GUI_Mode & selected_mode,
    int & selected_solver, VFRendering::View & vfr_view, bool & show_keybindings, bool & show_about,
    std::shared_ptr<State> state, std::vector<std::thread> & threads_image, std::thread & thread_chain );
void show_parameters( GUI_Mode & selected_mode );
void show_overlay_system( bool * p_open );
void show_overlay_calculation( bool * p_open, GUI_Mode & selected_mode, int & selected_solver );
void show_energy_plot();
void show_keybindings( bool & show_keybindings );
void show_about( bool & show_about );
void help_marker( const char * description );

} // namespace widgets

#endif