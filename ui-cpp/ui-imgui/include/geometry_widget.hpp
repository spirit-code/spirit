#pragma once
#ifndef SPIRIT_IMGUI_GEOMETRY_WIDGET_HPP
#define SPIRIT_IMGUI_GEOMETRY_WIDGET_HPP

#include <rendering_layer.hpp>
#include <widget_base.hpp>

#include <memory>

struct State;

namespace ui
{

struct GeometryWidget : public WidgetBase
{
    GeometryWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show_content() override;
    void update_data();

    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;

    int n_cells[3]{ 1, 1, 1 };
    int n_basis_atoms = 1;
    float bravais_vector_a[3]{ 0, 0, 0 };
    float bravais_vector_b[3]{ 0, 0, 0 };
    float bravais_vector_c[3]{ 0, 0, 0 };
    float lattice_constant = 1;

    int system_dimensionality;
    float system_size[3];
    float system_center[3];
    float system_bounds_min[3];
    float system_bounds_max[3];
    float cell_size[3];
    float cell_bounds_min[3];
    float cell_bounds_max[3];
};

} // namespace ui

#endif