#include <imgui_impl/fontawesome5_icons.h>
#include <visualisation_widget.hpp>
#include <widgets.hpp>

#include <Spirit/Geometry.h>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <fmt/format.h>

namespace ui
{

VisualisationWidget::VisualisationWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : WidgetBase( show ), state( state ), rendering_layer( rendering_layer )
{
    title    = "Visualisation settings";
    size_min = { 300, 300 };
    size_max = { 800, 999999 };
}

void VisualisationWidget::show_content()
{
    float * colour = rendering_layer.ui_shared_state.background_light.data();
    if( rendering_layer.ui_shared_state.dark_mode )
        colour = rendering_layer.ui_shared_state.background_dark.data();

    if( ImGui::Button( "Load settings" ) )
    {
    }
    ImGui::SameLine();
    if( ImGui::Button( "Save settings" ) )
    {
    }
    ImGui::SameLine();
    if( ImGui::Button( "Reset" ) )
    {
    }

    ImGui::Dummy( { 0, 10 } );

    ImGui::TextUnformatted( "Background color" );
    ImGui::SameLine();
    if( ImGui::Button( "default" ) )
    {
        if( rendering_layer.ui_shared_state.dark_mode )
            rendering_layer.ui_shared_state.background_dark = { 0.4f, 0.4f, 0.4f };
        else
            rendering_layer.ui_shared_state.background_light = { 0.9f, 0.9f, 0.9f };

        rendering_layer.set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>(
            glm::vec3{ colour[0], colour[1], colour[2] } );
    }

    if( ImGui::ColorEdit3( "##bgcolour", colour ) )
    {
        rendering_layer.set_view_option<VFRendering::View::Option::BACKGROUND_COLOR>(
            glm::vec3{ colour[0], colour[1], colour[2] } );
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Add Renderer" ) )
    {
        if( ImGui::IsPopupOpen( "##popup_add_renderer" ) )
            ImGui::CloseCurrentPopup();
        else
        {
            ImGui::OpenPopup( "##popup_add_renderer" );
        }
    }

    ImGui::Dummy( { 0, 10 } );

    if( ImGui::CollapsingHeader( "Overall filters" ) )
    {
        ImGui::Indent( 15 );

        ImGui::TextUnformatted( "Orientation" );
        ImGui::Indent( 15 );
        if( widgets::RangeSliderFloat(
                "##filter_direction_x", &rendering_layer.filter_direction_min[0],
                &rendering_layer.filter_direction_max[0], -1, 1, "x: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        if( widgets::RangeSliderFloat(
                "##filter_direction_y", &rendering_layer.filter_direction_min[1],
                &rendering_layer.filter_direction_max[1], -1, 1, "y: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        if( widgets::RangeSliderFloat(
                "##filter_direction_z", &rendering_layer.filter_direction_min[2],
                &rendering_layer.filter_direction_max[2], -1, 1, "z: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        ImGui::Indent( -15 );

        ImGui::TextUnformatted( "Position" );
        ImGui::Indent( 15 );
        if( widgets::RangeSliderFloat(
                "##filter_position_x", &rendering_layer.filter_position_min[0], &rendering_layer.filter_position_max[0],
                0, 1, "x: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        if( widgets::RangeSliderFloat(
                "##filter_position_y", &rendering_layer.filter_position_min[1], &rendering_layer.filter_position_max[1],
                0, 1, "y: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        if( widgets::RangeSliderFloat(
                "##filter_position_z", &rendering_layer.filter_position_min[2], &rendering_layer.filter_position_max[2],
                0, 1, "z: [%.3f, %.3f]" ) )
        {
            rendering_layer.update_visibility();
        }
        ImGui::Indent( -15 );

        ImGui::Indent( -15 );
    }

    if( ImGui::BeginPopup( "##popup_add_renderer" ) )
    {
        std::shared_ptr<ui::RendererWidget> renderer;
        if( ImGui::Selectable( "Dots" ) )
        {
            renderer = std::make_shared<ui::DotRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( ImGui::Selectable( "Arrows" ) )
        {
            renderer = std::make_shared<ui::ArrowRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( ImGui::Selectable( "Boxes" ) )
        {
            renderer = std::make_shared<ui::ParallelepipedRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( ImGui::Selectable( "Spheres" ) )
        {
            renderer = std::make_shared<ui::SphereRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( ImGui::Selectable( "Surface" ) )
        {
            renderer = std::make_shared<ui::SurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( ImGui::Selectable( "Isosurface" ) )
        {
            renderer = std::make_shared<ui::IsosurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                rendering_layer.vfr_update_deque );
        }
        if( renderer )
        {
            renderer->id = rendering_layer.renderer_id_counter;
            ++rendering_layer.renderer_id_counter;
            rendering_layer.renderer_widgets.push_back( renderer );
            rendering_layer.renderer_widgets_not_shown.push_back( renderer );
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    rendering_layer.coordinatesystem_renderer_widget->show();

    ImGui::Dummy( { 0, 10 } );

    rendering_layer.boundingbox_renderer_widget->show();

    for( auto & renderer_widget : rendering_layer.renderer_widgets )
    {
        ImGui::Dummy( { 0, 10 } );
        ImGui::Separator();
        ImGui::Dummy( { 0, 10 } );

        renderer_widget->show();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    ImGui::TextUnformatted( "Field of view" );
    if( ImGui::Button( ICON_FA_REDO ) )
    {
        rendering_layer.ui_shared_state.camera_perspective_fov = 45;
        rendering_layer.set_camera_fov( 45 );
    }
    ImGui::SameLine();
    if( ImGui::SliderInt( "##slider_fov", &rendering_layer.ui_shared_state.camera_perspective_fov, 0, 160 ) )
        rendering_layer.set_camera_fov( rendering_layer.ui_shared_state.camera_perspective_fov );

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    vgm::Vec3 dir(
        rendering_layer.ui_shared_state.light_direction[0], rendering_layer.ui_shared_state.light_direction[1],
        rendering_layer.ui_shared_state.light_direction[2] );
    bool update = false;
    ImGui::TextUnformatted( "Light direction" );
    ImGui::Columns( 2, "lightdircolumns", false ); // 3-ways, no border
    if( ImGui::gizmo3D( "##dir", dir ) )
        update = true;
    ImGui::NextColumn();
    auto normalize_light_dir = [&]()
    {
        auto norm = std::sqrt( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    };
    if( ImGui::InputFloat( "##lightdir_x", &dir.x, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_y", &dir.y, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_z", &dir.z, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( update )
    {
        normalize_light_dir();
        rendering_layer.ui_shared_state.light_direction = { dir.x, dir.y, dir.z };
        rendering_layer.set_view_option<VFRendering::View::Option::LIGHT_POSITION>(
            glm::vec3{ -1000 * dir.x, -1000 * dir.y, -1000 * dir.z } );
    }
    ImGui::Columns( 1 );
}

void VisualisationWidget::update_data() {}

} // namespace ui