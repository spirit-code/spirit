#include <visualisation_widget.hpp>

#include <imgui/imgui.h>

#include <imgui-gizmo3d/imGuIZMOquat.h>

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

    ImGui::TextUnformatted( "Renderers" );
    if( ImGui::Button( "Add Renderer" ) )
    {
        if( ImGui::IsPopupOpen( "##popup_add_renderer" ) )
            ImGui::CloseCurrentPopup();
        else
        {
            ImGui::OpenPopup( "##popup_add_renderer" );
        }
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