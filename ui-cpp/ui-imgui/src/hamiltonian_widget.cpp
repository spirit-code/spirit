#include <hamiltonian_widget.hpp>

#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>

#include <imgui/imgui.h>

#include <fmt/format.h>

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

HamiltonianWidget::HamiltonianWidget( bool & show, std::shared_ptr<State> state ) : show_( show ), state( state )
{
    this->update_data();
}

void HamiltonianWidget::show()
{
    if( !show_ )
        return;

    static std::vector<std::string> hamiltonian_types{ "Heisenberg", "Micromagnetic", "Gaussian" };
    static std::vector<std::string> ddi_methods{ "None", "FFT", "FMM", "Cutoff" };
    static int hamiltonian_type_idx = 0;
    auto & hamiltonian_type         = hamiltonian_types[hamiltonian_type_idx];

    ImGui::Begin( "Hamiltonian", &show_ );
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
            }

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Spin moment mu_s [mu_B]" );
            ImGui::Indent( 15 );
            ImGui::SetNextItemWidth( 80 );
            bool update_mu_s
                = ImGui::InputFloat( "##mu_s_0", &mu_s[0], 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue );
            for( int i = 1; i < mu_s.size() && i < 100; ++i )
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
            for( int i = 0; i < exchange.size() && i < 100; ++i )
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
            ImGui::Indent( -15 );
            if( exchange_active && update_exchange )
                Hamiltonian_Set_Exchange( state.get(), n_exchange_shells, exchange.data() );

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
            ImGui::Indent( -15 );
            if( dmi_active && update_dmi )
                Hamiltonian_Set_DMI( state.get(), n_dmi_shells, dmi.data(), dmi_chirality );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Dipolar interactions" );
            ImGui::Indent( 15 );
            ImGui::SetNextItemWidth( 220 );
            if( ImGui::BeginCombo( "##ddi_method", fmt::format( "Method: {}", ddi_methods[ddi_method] ).c_str() ) )
            {
                for( int idx = 0; idx < ddi_methods.size(); idx++ )
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
    ImGui::End();
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
    }
    else if( hamiltonian_type == "Gaussian" )
    {
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
    external_field_active = std::abs( external_field ) > 0;

    // Anisotropy
    Hamiltonian_Get_Anisotropy( state.get(), &anisotropy, anisotropy_dir.data() );
    anisotropy_active = std::abs( anisotropy ) > 0;

    // Exchange interaction (shells)
    exchange.reserve( 100 );
    Hamiltonian_Get_Exchange_Shells( state.get(), &n_exchange_shells, exchange.data() );
    exchange.resize( n_exchange_shells );
    exchange_active = n_exchange_shells > 0;

    // DMI
    // n_dmi_shells = Hamiltonian_Get_DMI_N_Pairs( state.get() );
    dmi.reserve( 100 );
    Hamiltonian_Get_DMI_Shells( state.get(), &n_dmi_shells, dmi.data(), &dmi_chirality );
    dmi.resize( n_dmi_shells );
    dmi_active = n_dmi_shells > 0;

    // DDI
    Hamiltonian_Get_DDI(
        state.get(), &ddi_method, ddi_n_periodic_images.data(), &ddi_cutoff_radius, &ddi_zero_padding );
    ddi_active = ddi_method != SPIRIT_DDI_METHOD_NONE;
}

} // namespace ui