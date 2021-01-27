#include <geometry_widget.hpp>

#include <Spirit/Geometry.h>

#include <imgui/imgui.h>

namespace ui
{

GeometryWidget::GeometryWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : show_( show ), state( state ), rendering_layer( rendering_layer )
{
    update_data();
}

void GeometryWidget::update_data()
{
    Geometry_Get_N_Cells( state.get(), n_cells );
    Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
    for( int dim = 0; dim < 3; ++dim )
        system_center[dim] = ( bounds_min[dim] + bounds_max[dim] ) / 2;
}

void GeometryWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Geometry", &show_ );

    ImGui::TextUnformatted( "Number of cells" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 160 );
    if( ImGui::InputInt3( "##geometry_n_cells", n_cells, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
        Geometry_Set_N_Cells( state.get(), n_cells );
        rendering_layer.update_vf_geometry();
        this->update_data();
    }

    ImGui::TextUnformatted( "Number of basis cell atoms" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 80 );
    if( ImGui::InputInt( "##geometry_n_basis_atoms", &n_basis_atoms, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
    }

    ImGui::TextUnformatted( "Bravais vectors" );
    ImGui::Indent( 15 );
    ImGui::TextUnformatted( "a" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 160 );
    if( ImGui::InputFloat3( "##geometry_bravais_vector_a", bravais_vector_a, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
    }
    ImGui::TextUnformatted( "b" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 160 );
    if( ImGui::InputFloat3( "##geometry_bravais_vector_b", bravais_vector_b, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
    }
    ImGui::TextUnformatted( "c" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 160 );
    if( ImGui::InputFloat3( "##geometry_bravais_vector_c", bravais_vector_c, ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
    }
    ImGui::Indent( -15 );

    ImGui::End();
}

} // namespace ui