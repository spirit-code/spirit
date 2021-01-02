#pragma once
#ifndef SPIRIT_IMGUI_GEOMETRY_WIDGET_HPP
#define SPIRIT_IMGUI_GEOMETRY_WIDGET_HPP

#include <rendering_layer.hpp>

#include <memory>

struct State;

namespace ui
{

struct GeometryWidget
{
    GeometryWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show();
    void update_data();

    bool & show_;
    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;

    int n_cells[3]{ 1, 1, 1 };
    int n_basis_atoms = 1;
    float bravais_vector_a[3]{ 0, 0, 0 };
    float bravais_vector_b[3]{ 0, 0, 0 };
    float bravais_vector_c[3]{ 0, 0, 0 };
};

} // namespace ui

#endif