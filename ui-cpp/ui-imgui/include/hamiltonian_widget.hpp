#pragma once
#ifndef SPIRIT_IMGUI_HAMILTONIAN_WIDGET_HPP
#define SPIRIT_IMGUI_HAMILTONIAN_WIDGET_HPP

#include <rendering_layer.hpp>
#include <widget_base.hpp>

#include <array>
#include <memory>
#include <vector>

struct State;

namespace ui
{

struct HamiltonianWidget : public WidgetBase
{
    HamiltonianWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void show_content() override;
    void update_data();
    void update_data_heisenberg();

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

    int exchange_n_pairs;
    std::vector<std::array<int, 2>> exchange_indices;
    std::vector<std::array<int, 3>> exchange_translations;
    std::vector<float> exchange_magnitudes;

    bool dmi_active;
    int n_dmi_shells;
    int dmi_chirality;
    std::vector<float> dmi;

    int dmi_n_pairs;
    std::vector<std::array<int, 2>> dmi_indices;
    std::vector<std::array<int, 3>> dmi_translations;
    std::vector<float> dmi_magnitudes;
    std::vector<std::array<float, 3>> dmi_normals;

    bool ddi_active;
    int ddi_method;
    std::array<int, 3> ddi_n_periodic_images;
    float ddi_cutoff_radius;
    bool ddi_zero_padding;
};

} // namespace ui

#endif