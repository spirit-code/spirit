#include <geometry_widget.hpp>

#include <Spirit/Geometry.h>

#include <imgui/imgui.h>

#include <fmt/format.h>

namespace ui
{

GeometryWidget::GeometryWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : WidgetBase( show ), state( state ), rendering_layer( rendering_layer )
{
    title = "Geometry";
    update_data();
}

void GeometryWidget::update_data()
{
    system_dimensionality = Geometry_Get_Dimensionality( state.get() );
    Geometry_Get_N_Cells( state.get(), n_cells );
    Geometry_Get_Center( this->state.get(), system_center );
    Geometry_Get_Bounds( state.get(), system_bounds_min, system_bounds_max );
    for( int dim = 0; dim < 3; ++dim )
        system_center[dim] = ( system_bounds_min[dim] + system_bounds_max[dim] ) / 2;

    Geometry_Get_Cell_Bounds( state.get(), cell_bounds_min, cell_bounds_max );
}

void GeometryWidget::show_content()
{
    static std::vector<std::string> bravais_lattice_types{
        "Irregular", "Rectilinear", "Simple cubic", "Hexagonal (2D)", "Hexagonal (2D, 60deg)", "Hexagonal (2D, 120deg)",
        "HCP",       "BCC",         "FCC",
    };
    static int bravais_lattice_type_idx = Bravais_Lattice_SC;

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

    ImGui::TextUnformatted( "Lattice constant" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 80 );
    if( ImGui::InputFloat(
            "##geometry_lattice_constant", &lattice_constant, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
    {
        Geometry_Set_Lattice_Constant( state.get(), lattice_constant );
    }
    ImGui::SameLine();
    ImGui::TextUnformatted( "[Å]" );

    ImGui::SetNextItemWidth( 280 );
    if( ImGui::BeginCombo(
            "##geometry_bravais_lattice",
            fmt::format( "Lattice type: {}", bravais_lattice_types[bravais_lattice_type_idx] ).c_str() ) )
    {
        for( int idx = 0; idx < bravais_lattice_types.size(); idx++ )
        {
            const bool is_selected = bravais_lattice_type_idx == idx;
            if( ImGui::Selectable( bravais_lattice_types[idx].c_str(), is_selected ) )
            {
                // bravais_lattice_type_idx = idx;
                // Geometry_Set_Bravais_Lattice_Type
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::TextUnformatted( "Bravais vectors" );
    ImGui::Indent( 15 );
    if( bravais_lattice_type_idx == Bravais_Lattice_Irregular )
    {
        ImGui::TextUnformatted( "a" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 160 );
        if( ImGui::InputFloat3(
                "##geometry_bravais_vector_a", bravais_vector_a, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            // Geometry_Set_Bravais_Vectors
        }
        ImGui::TextUnformatted( "b" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 160 );
        if( ImGui::InputFloat3(
                "##geometry_bravais_vector_b", bravais_vector_b, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            // Geometry_Set_Bravais_Vectors
        }
        ImGui::TextUnformatted( "c" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 160 );
        if( ImGui::InputFloat3(
                "##geometry_bravais_vector_c", bravais_vector_c, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            // Geometry_Set_Bravais_Vectors
        }
    }
    else
    {
        ImGui::TextUnformatted(
            fmt::format( "a ({} {} {})", bravais_vector_a[0], bravais_vector_a[1], bravais_vector_a[2] ).c_str() );
        ImGui::TextUnformatted(
            fmt::format( "b ({} {} {})", bravais_vector_b[0], bravais_vector_b[1], bravais_vector_b[2] ).c_str() );
        ImGui::TextUnformatted(
            fmt::format( "c ({} {} {})", bravais_vector_c[0], bravais_vector_c[1], bravais_vector_c[2] ).c_str() );
    }
    ImGui::Indent( -15 );

    // TODO:
    // - Geometry_Set_Cell_Atoms
    // - Geometry_Set_mu_s
    // - Geometry_Set_Cell_Atom_Types

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    ImGui::TextUnformatted( fmt::format( "Geometry is {}-dimensional", system_dimensionality ).c_str() );

    ImGui::Dummy( { 0, 10 } );

    for( int dim = 0; dim < 3; ++dim )
        cell_size[dim] = cell_bounds_max[dim] - cell_bounds_min[dim];
    ImGui::TextUnformatted( "Cell size" );
    ImGui::Indent( 15 );
    ImGui::TextUnformatted(
        fmt::format( "(x,y,z): {} x {} x {} [Å]", cell_size[0], cell_size[1], cell_size[2] ).c_str() );
    // TODO: translate into bravais basis extent
    // ImGui::TextUnformatted( fmt::format( "(a,b,c): {} x {} x {} [Å]", size[0], size[1], size[2] ).c_str() );
    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 0 } );

    for( int dim = 0; dim < 3; ++dim )
        system_size[dim] = system_bounds_max[dim] - system_bounds_min[dim];
    ImGui::TextUnformatted( "System size" );
    ImGui::Indent( 15 );
    ImGui::TextUnformatted(
        fmt::format( "(x,y,z): {} x {} x {} [Å]", system_size[0], system_size[1], system_size[2] ).c_str() );
    // TODO: translate into bravais basis extent
    // ImGui::TextUnformatted( fmt::format( "(a,b,c): {} x {} x {} [Å]", size[0], size[1], size[2] ).c_str() );
    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 0 } );

    ImGui::TextUnformatted( "System center" );
    ImGui::Indent( 15 );
    ImGui::TextUnformatted(
        fmt::format( "(x,y,z): {} x {} x {} [Å]", system_center[0], system_center[1], system_center[2] ).c_str() );
    ImGui::Indent( -15 );
}

} // namespace ui