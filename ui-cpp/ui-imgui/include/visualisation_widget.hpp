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
};

} // namespace ui

#endif