#pragma once
#ifndef SPIRIT_IMGUI_PLOTS_WIDGET_HPP
#define SPIRIT_IMGUI_PLOTS_WIDGET_HPP

#include <widget_base.hpp>

#include <Spirit/Spirit_Defines.h>

#include <memory>
#include <vector>

struct State;

namespace ui
{

struct PlotsWidget : public WidgetBase
{
    PlotsWidget( bool & show, std::shared_ptr<State> state );
    void show_content() override;
    void hook_pre_show() override;
    void update_data();

    std::shared_ptr<State> state;

    int force_index  = 0;
    int history_size = 200;
    std::vector<scalar> force_history;
    std::vector<scalar> iteration_history;
};

} // namespace ui

#endif