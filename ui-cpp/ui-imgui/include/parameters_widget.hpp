#pragma once
#ifndef SPIRIT_IMGUI_PARAMETERS_WIDGET_HPP
#define SPIRIT_IMGUI_PARAMETERS_WIDGET_HPP

#include <enums.hpp>
#include <rendering_layer.hpp>

#include <memory>

struct State;

namespace ui
{

struct ParametersWidget
{
    ParametersWidget( bool & show, std::shared_ptr<State> state, GUI_Mode & selected_mode );
    void show();
    void update_data();

    bool & show_;
    std::shared_ptr<State> state;
    GUI_Mode & selected_mode;

    float dt = 0;
};

} // namespace ui

#endif