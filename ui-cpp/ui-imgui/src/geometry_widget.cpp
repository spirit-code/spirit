#include <geometry_widget.hpp>

#include <Spirit/Geometry.h>

#include <imgui/imgui.h>

namespace ui
{

GeometryWidget::GeometryWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : show_( show ), state( state ), rendering_layer( rendering_layer )
{
    Geometry_Get_N_Cells( state.get(), n_cells );
}

void GeometryWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Geometry", &show_ );

    ImGui::TextUnformatted( "Number of cells" );
    ImGui::SameLine();
    if( ImGui::InputInt3( "##geometry_n_cells", n_cells, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
        Geometry_Set_N_Cells( state.get(), n_cells );
        rendering_layer.update_vf_geometry();
    }

    ImGui::End();
}

void GeometryWidget::update_data() {}

} // namespace ui