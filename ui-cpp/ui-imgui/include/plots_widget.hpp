#pragma once
#ifndef SPIRIT_IMGUI_PLOTS_WIDGET_HPP
#define SPIRIT_IMGUI_PLOTS_WIDGET_HPP

#include <memory>

struct State;

namespace ui
{

struct PlotsWidget
{
    PlotsWidget( bool & show, std::shared_ptr<State> state );
    void show();
    void update_data();

    bool & show_;
    std::shared_ptr<State> state;
};

} // namespace ui

#endif