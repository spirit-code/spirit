#pragma once
#ifndef SPIRIT_IMGUI_VISUALISATION_WIDGET_HPP
#define SPIRIT_IMGUI_VISUALISATION_WIDGET_HPP

#include <memory>
#include <rendering_layer.hpp>
#include <widget_base.hpp>

struct State;

namespace ui
{

struct VisualisationWidget : public WidgetBase
{
    VisualisationWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show_content() override;
    void update_data();

    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;
};

} // namespace ui

#endif