#pragma once
#ifndef SPIRIT_IMGUI_VFR_LAYER_HPP
#define SPIRIT_IMGUI_VFR_LAYER_HPP

#include <enums.hpp>
#include <renderer_widget.hpp>
#include <ui_shared_state.hpp>

#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/View.hxx>

#include <Spirit/Simulation.h>

#include <deque>
#include <functional>
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
    // Change the field of view without changing the perceived camera distance
    void set_camera_fov( float new_fov );
    void set_camera_orthographic( bool orthographic );

    void set_interaction_mode( UiSharedState::InteractionMode interaction_mode );

    void screenshot_png( std::string filename = "screenshot" );

    void update_theme();
    void update_visibility();
    void update_renderers();
    void update_renderers_from_layout();
    void update_boundingbox();
    void update_vf_geometry();
    void update_vf_directions();

    void needs_redraw();
    void needs_data();

    template<int OptionIndex, typename T2>
    void set_view_option( const T2 & arg )
    {
        this->vfr_update_deque.push_back( [&, arg]() { this->view.setOption<OptionIndex>( arg ); } );
    }

    // Visualisation Settings
    std::shared_ptr<State> state;
    VFRendering::View view;
    UiSharedState & ui_shared_state;
    VFRendering::VectorField vectorfield = VFRendering::VectorField( {}, {} );
    std::deque<std::function<void()>> vfr_update_deque;

    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets;
    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets_not_shown;
    std::shared_ptr<BoundingBoxRendererWidget> boundingbox_renderer_widget;
    std::shared_ptr<CoordinateSystemRendererWidget> coordinatesystem_renderer_widget;

    int n_cell_step         = 1;
    int renderer_id_counter = 0;

    float filter_direction_min[3]{ -1, -1, -1 };
    float filter_direction_max[3]{ 1, 1, 1 };
    float filter_position_min[3]{ 0, 0, 0 };
    float filter_position_max[3]{ 1, 1, 1 };

    std::array<float, 4> rendering_layout = { 0, 0, 1, 1 };

private:
    std::vector<std::shared_ptr<RendererWidget>> renderer_widgets_shown;

    bool gl_initialized_ = false;
    bool needs_redraw_   = true;
    bool needs_data_     = true;

    bool show_coordinatesystem_ = true;
    bool show_boundingbox_      = true;

    VFRendering::Geometry geometry;
    VFRendering::VectorField vectorfield_surf2D = VFRendering::VectorField( {}, {} );
    std::shared_ptr<VFRendering::CombinedRenderer> combined_renderer;
};

} // namespace ui

#endif