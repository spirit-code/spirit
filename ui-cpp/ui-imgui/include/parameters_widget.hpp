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

    float llg_dt          = 0;
    float llg_convergence = 0;

    float gneb_convergence         = 1e-8;
    int gneb_image_type            = 0;
    float gneb_spring_constant     = 1;
    float gneb_spring_energy_ratio = 0;
    float gneb_path_shortening     = 0;
};

} // namespace ui

#endif