#pragma once
#ifndef SPIRIT_IMGUI_WIDGETS_HPP
#define SPIRIT_IMGUI_WIDGETS_HPP

#include <GLFW/glfw3.h>

#include <VFRendering/View.hxx>

namespace widgets
{

void show_menu_bar(
    GLFWwindow * window, ImFont * font, bool & dark_mode, ImVec4 & background_colour, int & selected_mode,
    VFRendering::View & vfr_view );
void help_marker( const char * description );
void show_overlay( bool * p_open );

} // namespace widgets

#endif