#pragma once
#ifndef SPIRIT_IMGUI_WIDGETS_HPP
#define SPIRIT_IMGUI_WIDGETS_HPP

#include <enums.hpp>

#include <rendering_layer.hpp>
#include <settings.hpp>

#include <thread>
#include <vector>

struct State;

namespace widgets
{

void help_marker( const char * description );

void show_parameters( bool & show, GUI_Mode & selected_mode );
void show_plots( bool & show );
void show_visualisation_settings( bool & show, ui::RenderingLayer & rendering_layer );
void show_settings( bool & show, ui::RenderingLayer & rendering_layer );

void show_overlay_system( bool & show );
void show_overlay_calculation(
    bool & show, GUI_Mode & selected_mode, int & selected_solver_min, int & selected_solver_llg );

void show_keybindings( bool & show );
void show_about( bool & show );

} // namespace widgets

#endif