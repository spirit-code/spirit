#pragma once
#ifndef SPIRIT_IMGUI_WIDGETS_HPP
#define SPIRIT_IMGUI_WIDGETS_HPP

#include <enums.hpp>
#include <rendering_layer.hpp>

#include <thread>
#include <vector>

struct State;

namespace widgets
{

bool toggle_button( const char * str_id, bool * v, bool coloured = true );
void help_marker( const char * description );

void show_overlay_system( bool & show, int & corner, std::array<float, 2> & position, std::shared_ptr<State> state );
void show_overlay_calculation(
    bool & show, GUI_Mode & selected_mode, int & selected_solver_min, int & selected_solver_llg, int & corner,
    std::array<float, 2> & position, std::shared_ptr<State> state );

void show_settings( bool & show, ui::RenderingLayer & rendering_layer );

void show_keybindings( bool & show );
void show_about( bool & show );

} // namespace widgets

#endif