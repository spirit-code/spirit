#pragma once
#ifndef SPIRIT_IMGUI_VFR_LAYER_HPP
#define SPIRIT_IMGUI_VFR_LAYER_HPP

#include <enums.hpp>
#include <renderer_widget.hpp>
#include <ui_shared_state.hpp>

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
    RenderingLayer( UiSharedState & ui_shared_state, std::shared_ptr<State> state );

    void initialize_gl();
    void draw( int display_w, int display_h );
    void reset_camera();
    void screenshot_png( std::string filename = "screenshot" );

    void update_vf_geometry();
    void update_vf_directions();

    void needs_redraw();
    void needs_data();

    // Visualisation Settings
    std::shared_ptr<State> state;
    VFRendering::View view;
    UiSharedState & ui_shared_state;
    VFRendering::VectorField vectorfield = VFRendering::VectorField( {}, {} );

    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets;
    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets_not_shown;

    int n_cell_step = 1;

private:
    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets_shown;

    bool gl_initialized_ = false;
    bool needs_redraw_   = true;
    bool needs_data_     = true;

    VFRendering::Geometry geometry;
    VFRendering::VectorField vectorfield_surf2D = VFRendering::VectorField( {}, {} );

    std::shared_ptr<VFRendering::ArrowRenderer> arrow_renderer_ptr;
    std::vector<std::shared_ptr<VFRendering::RendererBase>> system_renderers{};
};

} // namespace ui

#endif