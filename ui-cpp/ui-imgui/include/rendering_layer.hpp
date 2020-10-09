#pragma once
#ifndef SPIRIT_IMGUI_VFR_LAYER_HPP
#define SPIRIT_IMGUI_VFR_LAYER_HPP

#include <enums.hpp>
#include <settings.hpp>

#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/View.hxx>

#include <Spirit/Simulation.h>

#include <memory>
#include <thread>
#include <vector>

struct State;

namespace ui
{

struct RenderingLayer
{
    RenderingLayer( std::shared_ptr<ui::Settings> settings, std::shared_ptr<State> state );

    void initialize_gl();
    void draw( int display_w, int display_h );
    void screenshot_png( std::string filename = "screenshot" );

    void update_vf_geometry();
    void update_vf_directions();

    void needs_redraw();
    void needs_data();

    // Visualisation Settings
    VFRendering::View view;
    std::shared_ptr<ui::Settings> settings;

    glm::vec4 background_colour_dark  = glm::vec4{ 0.4f, 0.4f, 0.4f, 0.f };
    glm::vec4 background_colour_light = glm::vec4{ 0.7f, 0.7f, 0.7f, 0.f };

    int n_cell_step = 1;

private:
    std::shared_ptr<State> state;

    bool gl_initialized_ = false;
    bool needs_redraw_   = true;
    bool needs_data_     = true;

    VFRendering::Geometry geometry;
    VFRendering::VectorField vectorfield        = VFRendering::VectorField( {}, {} );
    VFRendering::VectorField vectorfield_surf2D = VFRendering::VectorField( {}, {} );

    std::shared_ptr<VFRendering::ArrowRenderer> arrow_renderer_ptr;
    std::vector<std::shared_ptr<VFRendering::RendererBase>> system_renderers{};
};

} // namespace ui

#endif