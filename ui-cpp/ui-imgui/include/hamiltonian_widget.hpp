#pragma once
#ifndef SPIRIT_IMGUI_HAMILTONIAN_WIDGET_HPP
#define SPIRIT_IMGUI_HAMILTONIAN_WIDGET_HPP

#include <rendering_layer.hpp>

#include <array>
#include <memory>
#include <vector>

struct State;

namespace ui
{

struct HamiltonianWidget
{
    HamiltonianWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show();
    void update_data();
    void update_data_heisenberg();

    bool & show_;
    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;

    std::array<bool, 3> boundary_conditions;

    std::vector<float> mu_s;

    bool external_field_active;
    float external_field;
    std::array<float, 3> external_field_dir;

    bool anisotropy_active;
    float anisotropy;
    std::array<float, 3> anisotropy_dir;

    bool exchange_active;
    int n_exchange_shells;
    std::vector<float> exchange;

    bool dmi_active;
    int n_dmi_shells;
    int dmi_chirality;
    std::vector<float> dmi;

    bool ddi_active;
    int ddi_method;
    std::array<int, 3> ddi_n_periodic_images;
    float ddi_cutoff_radius;
    bool ddi_zero_padding;
};

} // namespace ui

#endif