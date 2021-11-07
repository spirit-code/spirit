#include <fonts.hpp>
#include <hamiltonian_widget.hpp>

#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>

#include <imgui/imgui.h>

#include <fmt/format.h>

#include <array>
#include <string>
#include <vector>

namespace ui
{

void normalize( std::array<float, 3> vec )
{
    float x2 = vec[0] * vec[0];
    float y2 = vec[1] * vec[1];
    float z2 = vec[2] * vec[2];
    float v  = std::sqrt( x2 + y2 + z2 );
    if( v > 0 )
    {
        vec[0] /= v;
        vec[1] /= v;
        vec[2] /= v;
    }
}

HamiltonianWidget::HamiltonianWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : WidgetBase( show ), state( state ), rendering_layer( rendering_layer )
{
    title = "Hamiltonian";
    this->update_data();
}

void HamiltonianWidget::show_content()
{
    static std::vector<std::string> hamiltonian_types{ "Heisenberg", "Micromagnetic", "Gaussian" };
    static std::vector<std::string> ddi_methods{ "None", "FFT", "FMM", "Cutoff" };
    static int hamiltonian_type_idx = 0;
    auto & hamiltonian_type         = hamiltonian_types[hamiltonian_type_idx];

    {
        ImGui::SetNextItemWidth( 220 );
        if( ImGui::BeginCombo(
                "##hamiltonian_type",
                fmt::format( "{} Hamiltonian", hamiltonian_types[hamiltonian_type_idx] ).c_str() ) )
        {
            for( int idx = 0; idx < hamiltonian_types.size(); idx++ )
            {
                const bool is_selected = hamiltonian_type_idx == idx;
                if( ImGui::Selectable( hamiltonian_types[idx].c_str(), is_selected ) )
                {
                    hamiltonian_type_idx = idx;
                    // TODO: set the Hamiltonian
                }

                // // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                // if( is_selected )
                //     ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if( hamiltonian_type == "Heisenberg" )
        {
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "Periodical boundary conditions" );
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "(a, b, c)" );
            ImGui::SameLine();
            bool update_bc = false;
            if( ImGui::Checkbox( "##periodical_a", &boundary_conditions[0] ) )
                update_bc = true;
            ImGui::SameLine();
            if( ImGui::Checkbox( "##periodical_b", &boundary_conditions[1] ) )
                update_bc = true;
            ImGui::SameLine();
            if( ImGui::Checkbox( "##periodical_c", &boundary_conditions[2] ) )
                update_bc = true;
            ImGui::Indent( -15 );
            ImGui::Indent( -15 );
            if( update_bc )
            {
                Hamiltonian_Set_Boundary_Conditions( state.get(), boundary_conditions.data() );
                rendering_layer.update_boundingbox();
            }

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Spin moment mu_s [mu_B]" );
            ImGui::Indent( 15 );
            ImGui::SetNextItemWidth( 80 );
            bool update_mu_s
                = ImGui::InputFloat( "##mu_s_0", &mu_s[0], 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue );
            for( std::size_t i = 1; i < mu_s.size() && i < 100; ++i )
            {
                // Up to 3 in a row
                if( i % 3 != 0 )
                    ImGui::SameLine();
                ImGui::SetNextItemWidth( 80 );
                update_mu_s = update_mu_s
                              || ImGui::InputFloat(
                                  fmt::format( "##mu_s_{}", i ).c_str(), &mu_s[i], 0, 0, "%.3f",
                                  ImGuiInputTextFlags_EnterReturnsTrue );
            }
            ImGui::SetNextItemWidth( 120 );
            // ImGui::InputScalarN( "##mu_s", ImGuiDataType_Float, mu_s.data(), mu_s.size() )
            if( update_mu_s )
            {
                // TODO: add API to set all spin moments of the basis cell
                Geometry_Set_mu_s( state.get(), mu_s[0] );
            }
            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            bool update_external_field = false;
            if( ImGui::Checkbox( "External field", &external_field_active ) )
            {
                if( external_field_active )
                    update_external_field = true;
                else
                    Hamiltonian_Set_Field( state.get(), 0, external_field_dir.data() );
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 100 );
            if( ImGui::InputFloat(
                    "##external_field", &external_field, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
                update_external_field = true;
            ImGui::SameLine();
            ImGui::TextUnformatted( "T" );
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "dir" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 140 );
            if( ImGui::InputFloat3(
                    "##external_field_dir", external_field_dir.data(), "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                update_external_field = true;
                normalize( external_field_dir );
            }
            ImGui::Indent( -15 );
            if( external_field_active && update_external_field )
                Hamiltonian_Set_Field( state.get(), external_field, external_field_dir.data() );

            ImGui::Dummy( { 0, 10 } );

            bool update_anisotropy = true;
            if( ImGui::Checkbox( "Anisotropy", &anisotropy_active ) )
            {
                if( anisotropy_active )
                    update_anisotropy = true;
                else
                    Hamiltonian_Set_Anisotropy( state.get(), 0, anisotropy_dir.data() );
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 100 );
            if( ImGui::InputFloat( "##anisotropy", &anisotropy, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
                update_anisotropy = true;
            ImGui::SameLine();
            ImGui::TextUnformatted( "T" );
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "dir" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 140 );
            if( ImGui::InputFloat3(
                    "##anisotropy_dir", anisotropy_dir.data(), "%.3f", ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                update_anisotropy = true;
                normalize( anisotropy_dir );
            }
            ImGui::Indent( -15 );
            if( anisotropy_active && update_anisotropy )
                Hamiltonian_Set_Anisotropy( state.get(), anisotropy, anisotropy_dir.data() );

            ImGui::Dummy( { 0, 10 } );

            bool update_exchange = false;
            if( ImGui::Checkbox( "Exchange interaction", &exchange_active ) )
            {
                if( exchange_active )
                    update_exchange = true;
                else
                    Hamiltonian_Set_Exchange( state.get(), 0, nullptr, 1 );
            }
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "Number of shells" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 100 );
            if( ImGui::InputInt(
                    "##n_exchange_shells", &n_exchange_shells, 1, 5, ImGuiInputTextFlags_EnterReturnsTrue ) )
            {
                exchange.resize( n_exchange_shells, 0 );
                update_exchange = true;
            }
            ImGui::SetNextItemWidth( 140 );
            // ImGui::InputScalarN( "##exchange_magnitudes", ImGuiDataType_Float, exchange.data(), exchange.size() );
            for( std::size_t i = 0; i < exchange.size() && i < 100; ++i )
            {
                // Up to 3 in a row
                if( i % 3 != 0 )
                    ImGui::SameLine();
                ImGui::SetNextItemWidth( 80 );
                if( ImGui::InputFloat(
                        fmt::format( "##exchange_{}", i ).c_str(), &exchange[i], 0, 0, "%.5f",
                        ImGuiInputTextFlags_EnterReturnsTrue ) )
                    update_exchange = true;
            }

            if( exchange_active && update_exchange )
            {
                Hamiltonian_Set_Exchange( state.get(), n_exchange_shells, exchange.data() );
                update_exchange = false;
                this->update_data();
            }

            if( ImGui::TreeNode( "Exchange pairs" ) )
            {
                if( exchange_n_pairs > 0
                    && ImGui::BeginTable(
                        "table_exchange", 6, ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_SizingFixedFit ) )
                {
                    ImGui::TableSetupColumn( "i", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "j", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "da", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "db", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "dc", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "Jij", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableHeadersRow();

                    for( int row = 0; row < exchange_n_pairs; row++ )
                    {
                        ImGui::TableNextRow();

                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##exchange_i_{}", row ).c_str(), &exchange_indices[row][0], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##exchange_j_{}", row ).c_str(), &exchange_indices[row][1], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##exchange_da_{}", row ).c_str(), &exchange_translations[row][0], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##exchange_db_{}", row ).c_str(), &exchange_translations[row][1], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##exchange_dc_{}", row ).c_str(), &exchange_translations[row][2], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 80 );
                        if( ImGui::InputFloat(
                                fmt::format( "##exchange_jij_{}", row ).c_str(), &exchange_magnitudes[row], 0, 0,
                                "%.5f", ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_exchange = true;
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::Button( ICON_FA_PLUS "  add pair##add_exchange_pair" );
            }

            if( exchange_active && update_exchange )
            {
                // TODO: add API to set Exchange pairs
            }

            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            bool update_dmi = false;
            if( ImGui::Checkbox( "Dzyaloshinskii-Moriya interaction", &dmi_active ) )
            {
                if( dmi_active )
                    update_dmi = true;
                else
                    Hamiltonian_Set_DMI( state.get(), 0, nullptr, 1 );
            }
            ImGui::Indent( 15 );
            ImGui::TextUnformatted( "Number of shells" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 100 );
            ImGui::InputInt( "##n_dmi_shells", &n_dmi_shells, 1, 5, ImGuiInputTextFlags_EnterReturnsTrue );
            ImGui::SetNextItemWidth( 140 );
            // ImGui::InputScalarN(
            //     "##dmi_magnitudes", ImGuiDataType_Float, dmi.data(), dmi.size(), nullptr, nullptr, nullptr,
            //     ImGuiInputTextFlags_EnterReturnsTrue );
            for( int i = 0; i < n_dmi_shells; ++i )
            {
                // Up to 3 in a row
                if( i % 3 != 0 )
                    ImGui::SameLine();
                if( ImGui::InputFloat(
                        fmt::format( "##dmi_{}", i ).c_str(), &dmi[i], 0, 0, "%.5f",
                        ImGuiInputTextFlags_EnterReturnsTrue ) )
                    update_dmi = true;
            }

            if( dmi_active && update_dmi )
            {
                Hamiltonian_Set_DMI( state.get(), n_dmi_shells, dmi.data(), dmi_chirality );
                update_dmi = false;
            }

            if( ImGui::TreeNode( "DMI pairs" ) )
            {
                // TODO: add API to get DMI pairs

                ImGui::TextUnformatted( "DMI pairs are not yet available" );

                if( dmi_n_pairs > 0
                    && ImGui::BeginTable(
                        "table_dmi", 7, ImGuiTableFlags_NoPadInnerX | ImGuiTableFlags_SizingFixedFit ) )
                {
                    ImGui::TableSetupColumn( "i", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "j", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "da", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "db", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "dc", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "Dij", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableSetupColumn( "DM normal", ImGuiTableColumnFlags_WidthFixed );
                    ImGui::TableHeadersRow();

                    for( int row = 0; row < dmi_n_pairs; row++ )
                    {
                        ImGui::TableNextRow();

                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##dmi_i_{}", row ).c_str(), &dmi_indices[row][0], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##dmi_j_{}", row ).c_str(), &dmi_indices[row][1], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##dmi_da_{}", row ).c_str(), &dmi_translations[row][0], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##dmi_db_{}", row ).c_str(), &dmi_translations[row][1], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 40 );
                        if( ImGui::InputInt(
                                fmt::format( "##dmi_dc_{}", row ).c_str(), &dmi_translations[row][2], 0, 0,
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 80 );
                        if( ImGui::InputFloat(
                                fmt::format( "##dmi_dij_{}", row ).c_str(), &dmi_magnitudes[row], 0, 0, "%.5f",
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::SetNextItemWidth( 140 );
                        if( ImGui::InputFloat3(
                                fmt::format( "##dmi_normal_{}", row ).c_str(), &dmi_normals[row][0], "%.3f",
                                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ReadOnly ) )
                        {
                            update_dmi = true;
                        }
                    }
                    ImGui::EndTable();
                }

                ImGui::Button( ICON_FA_PLUS "  add pair##add_dmi_pair" );
            }

            if( dmi_active && update_dmi )
            {
                // TODO: add API to set DMI pairs
            }

            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Dipolar interactions" );
            ImGui::Indent( 15 );
            ImGui::SetNextItemWidth( 220 );
            if( ImGui::BeginCombo( "##ddi_method", fmt::format( "Method: {}", ddi_methods[ddi_method] ).c_str() ) )
            {
                for( std::size_t idx = 0; idx < ddi_methods.size(); idx++ )
                {
                    const bool is_selected = ddi_method == idx;
                    if( ImGui::Selectable( ddi_methods[idx].c_str(), is_selected ) )
                    {
                        ddi_method = idx;
                        // TODO: set
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextUnformatted( "Periodic images" );
            ImGui::Indent( 15 );
            ImGui::SetNextItemWidth( 140 );
            ImGui::InputInt3(
                "##ddi_n_periodic_images", ddi_n_periodic_images.data(), ImGuiInputTextFlags_EnterReturnsTrue );
            ImGui::Indent( -15 );
            ImGui::TextUnformatted( "Cutoff radius" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            ImGui::InputFloat(
                "##ddi_cutoff_radius", &ddi_cutoff_radius, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue );
            ImGui::Checkbox( "zero-padding", &ddi_zero_padding );
            ImGui::Indent( -15 );
        }
        else if( hamiltonian_type == "Micromagnetic" )
        {
        }
        else if( hamiltonian_type == "Gaussian" )
        {
        }
    }
}

void HamiltonianWidget::update_data()
{
    std::string hamiltonian_type = Hamiltonian_Get_Name( state.get() );
    if( hamiltonian_type == "Heisenberg" )
    {
        this->update_data_heisenberg();
    }
    else if( hamiltonian_type == "Micromagnetic" )
    {
        // TODO
    }
    else if( hamiltonian_type == "Gaussian" )
    {
        // TODO
    }
}

void HamiltonianWidget::update_data_heisenberg()
{
    int n_basis_atoms = Geometry_Get_N_Cell_Atoms( state.get() );
    mu_s.resize( n_basis_atoms );

    // Boundary conditions
    Hamiltonian_Get_Boundary_Conditions( state.get(), boundary_conditions.data() );

    // mu_s
    Geometry_Get_mu_s( state.get(), mu_s.data() );

    // External magnetic field
    Hamiltonian_Get_Field( state.get(), &external_field, external_field_dir.data() );

    // Anisotropy
    Hamiltonian_Get_Anisotropy( state.get(), &anisotropy, anisotropy_dir.data() );

    // Exchange interaction (shells)
    exchange.resize( 100 );
    Hamiltonian_Get_Exchange_Shells( state.get(), &n_exchange_shells, exchange.data() );
    exchange.resize( n_exchange_shells );

    // Exchange interaction (pairs)
    exchange_n_pairs      = Hamiltonian_Get_Exchange_N_Pairs( state.get() );
    exchange_indices      = std::vector<std::array<int, 2>>( exchange_n_pairs );
    exchange_translations = std::vector<std::array<int, 3>>( exchange_n_pairs );
    exchange_magnitudes   = std::vector<float>( exchange_n_pairs, 0 );
    Hamiltonian_Get_Exchange_Pairs(
        state.get(), reinterpret_cast<int( * )[2]>( exchange_indices[0].data() ),
        reinterpret_cast<int( * )[3]>( exchange_translations.data() ), exchange_magnitudes.data() );

    // DMI (shells)
    dmi.resize( 100 );
    Hamiltonian_Get_DMI_Shells( state.get(), &n_dmi_shells, dmi.data(), &dmi_chirality );
    dmi.resize( n_dmi_shells );
    dmi_active = n_dmi_shells > 0;

    // DMI (pairs)
    dmi_n_pairs      = Hamiltonian_Get_DMI_N_Pairs( state.get() );
    dmi_indices      = std::vector<std::array<int, 2>>( dmi_n_pairs );
    dmi_translations = std::vector<std::array<int, 3>>( dmi_n_pairs );
    dmi_magnitudes   = std::vector<float>( dmi_n_pairs, 0 );
    dmi_normals      = std::vector<std::array<float, 3>>( dmi_n_pairs );
    // Hamiltonian_Get_DMI_Pairs(
    //     state.get(), (int( * )[2])dmi_indices.data(), (int( * )[3])dmi_translations.data(), dmi_magnitudes.data(),
    //     (float( * )[3])dmi_normals );

    // DDI
    Hamiltonian_Get_DDI(
        state.get(), &ddi_method, ddi_n_periodic_images.data(), &ddi_cutoff_radius, &ddi_zero_padding );

    // Which are active
    external_field_active = std::abs( external_field ) > 0;
    anisotropy_active     = std::abs( anisotropy ) > 0;
    exchange_active       = exchange_n_pairs > 0;
    dmi_active            = dmi_n_pairs > 0;
    ddi_active            = ddi_method != SPIRIT_DDI_METHOD_NONE;
}

} // namespace ui