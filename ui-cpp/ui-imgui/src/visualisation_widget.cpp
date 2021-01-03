#include <visualisation_widget.hpp>
#include <widgets.hpp>

#include <Spirit/Geometry.h>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

#include <fmt/format.h>

namespace ui
{

VisualisationWidget::VisualisationWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : show_( show ), state( state ), rendering_layer( rendering_layer )
{
}

void VisualisationWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Visualisation settings", &show_ );

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

        rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { colour[0], colour[1], colour[2] } );
    }

    if( ImGui::ColorEdit3( "##bgcolour", colour ) )
    {
        rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { colour[0], colour[1], colour[2] } );
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

    auto is_visible = [&]() -> std::string {
        const float epsilon = 1e-5;

        float b_min[3], b_max[3], b_range[3];
        Geometry_Get_Bounds( state.get(), b_min, b_max );

        float filter_pos_min[3], filter_pos_max[3];
        float filter_dir_min[3], filter_dir_max[3];
        for( int dim = 0; dim < 3; ++dim )
        {
            b_range[dim]        = b_max[dim] - b_min[dim];
            filter_pos_min[dim] = b_min[dim] + filter_position_min[dim] * b_range[dim] - epsilon;
            filter_pos_max[dim] = b_max[dim] + ( filter_position_max[dim] - 1 ) * b_range[dim] + epsilon;

            filter_dir_min[dim] = filter_direction_min[dim] - epsilon;
            filter_dir_max[dim] = filter_direction_max[dim] + epsilon;
        }
        return fmt::format(
            R"(
            bool is_visible(vec3 position, vec3 direction)
            {{
                float x_min_pos = {};
                float x_max_pos = {};
                bool is_visible_x_pos = position.x <= x_max_pos && position.x >= x_min_pos;

                float y_min_pos = {};
                float y_max_pos = {};
                bool is_visible_y_pos = position.y <= y_max_pos && position.y >= y_min_pos;

                float z_min_pos = {};
                float z_max_pos = {};
                bool is_visible_z_pos = position.z <= z_max_pos && position.z >= z_min_pos;

                float x_min_dir = {};
                float x_max_dir = {};
                bool is_visible_x_dir = direction.x <= x_max_dir && direction.x >= x_min_dir;

                float y_min_dir = {};
                float y_max_dir = {};
                bool is_visible_y_dir = direction.y <= y_max_dir && direction.y >= y_min_dir;

                float z_min_dir = {};
                float z_max_dir = {};
                bool is_visible_z_dir = direction.z <= z_max_dir && direction.z >= z_min_dir;

                return is_visible_x_pos && is_visible_y_pos && is_visible_z_pos && is_visible_x_dir && is_visible_y_dir && is_visible_z_dir;
            }}
            )",
            filter_pos_min[0], filter_pos_max[0], filter_pos_min[1], filter_pos_max[1], filter_pos_min[2],
            filter_pos_max[2], filter_direction_min[0], filter_direction_max[0], filter_direction_min[1],
            filter_direction_max[1], filter_direction_min[2], filter_direction_max[2] );
    };

    if( ImGui::CollapsingHeader( "Overall filters" ) )
    {
        ImGui::Indent( 15 );

        ImGui::TextUnformatted( "Orientation" );
        ImGui::Indent( 15 );
        if( widgets::RangeSliderFloat(
                "##filter_direction_x", &filter_direction_min[0], &filter_direction_max[0], -1, 1, "x: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
        }
        if( widgets::RangeSliderFloat(
                "##filter_direction_y", &filter_direction_min[1], &filter_direction_max[1], -1, 1, "y: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
        }
        if( widgets::RangeSliderFloat(
                "##filter_direction_z", &filter_direction_min[2], &filter_direction_max[2], -1, 1, "z: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
        }
        ImGui::Indent( -15 );

        ImGui::TextUnformatted( "Position" );
        ImGui::Indent( 15 );
        if( widgets::RangeSliderFloat(
                "##filter_position_x", &filter_position_min[0], &filter_position_max[0], 0, 1, "x: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
        }
        if( widgets::RangeSliderFloat(
                "##filter_position_y", &filter_position_min[1], &filter_position_max[1], 0, 1, "y: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
        }
        if( widgets::RangeSliderFloat(
                "##filter_position_z", &filter_position_min[2], &filter_position_max[2], 0, 1, "z: [%.3f, %.3f]" ) )
        {
            rendering_layer.view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( is_visible() );
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
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Arrows" ) )
        {
            renderer = std::make_shared<ui::ArrowRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Boxes" ) )
        {
            renderer = std::make_shared<ui::ParallelepipedRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Spheres" ) )
        {
            renderer = std::make_shared<ui::SphereRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Surface" ) )
        {
            renderer = std::make_shared<ui::SurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( ImGui::Selectable( "Isosurface" ) )
        {
            renderer = std::make_shared<ui::IsosurfaceRendererWidget>(
                rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
        }
        if( renderer )
        {
            rendering_layer.renderer_widgets.push_back( renderer );
            rendering_layer.renderer_widgets_not_shown.push_back( renderer );
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

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

    vgm::Vec3 dir(
        rendering_layer.ui_shared_state.light_direction[0], rendering_layer.ui_shared_state.light_direction[1],
        rendering_layer.ui_shared_state.light_direction[2] );
    bool update = false;
    ImGui::TextUnformatted( "Light direction" );
    ImGui::Columns( 2, "lightdircolumns", false ); // 3-ways, no border
    if( ImGui::gizmo3D( "##dir", dir ) )
        update = true;
    ImGui::NextColumn();
    auto normalize_light_dir = [&]() {
        auto norm = std::sqrt( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    };
    if( ImGui::InputFloat( "##lightdir_x", &dir.x, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_y", &dir.y, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( ImGui::InputFloat( "##lightdir_z", &dir.z, 0, 0, 3, ImGuiInputTextFlags_EnterReturnsTrue ) )
        update = true;
    if( update )
    {
        normalize_light_dir();
        rendering_layer.ui_shared_state.light_direction = { dir.x, dir.y, dir.z };
        rendering_layer.view.setOption<VFRendering::View::Option::LIGHT_POSITION>(
            { -1000 * dir.x, -1000 * dir.y, -1000 * dir.z } );
    }

    ImGui::Columns( 1 );

    ImGui::End();
}

void VisualisationWidget::update_data() {}

} // namespace ui