#pragma once
#ifndef SPIRIT_IMGUI_VISUALISATION_WIDGET_HPP
#define SPIRIT_IMGUI_VISUALISATION_WIDGET_HPP

#include <rendering_layer.hpp>

#include <memory>

struct State;

namespace ui
{

struct VisualisationWidget
{
    VisualisationWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show();
    void update_data();

    bool & show_;
    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;

    float filter_direction_min[3]{ -1, -1, -1 };
    float filter_direction_max[3]{ 1, 1, 1 };
    float filter_position_min[3]{ 0, 0, 0 };
    float filter_position_max[3]{ 1, 1, 1 };
};

} // namespace ui

#endif