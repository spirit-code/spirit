#include <parameters_widget.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Parameters_EMA.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Parameters_MC.h>
#include <Spirit/Parameters_MMF.h>

#include <imgui/imgui.h>
#include <imgui/misc/cpp/imgui_stdlib.h>

namespace ui
{

ParametersWidget::ParametersWidget( bool & show, std::shared_ptr<State> state, UiSharedState & ui_shared_state )
        : WidgetBase( show ), state( state ), ui_shared_state( ui_shared_state )
{
    title = "Parameters";
    update_data();
}

void ParametersWidget::show_content()
{
    if( ui_shared_state.selected_mode == GUI_Mode::Minimizer )
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.min_apply_to_all );
    else if( ui_shared_state.selected_mode == GUI_Mode::LLG )
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.llg_apply_to_all );
    if( ui_shared_state.selected_mode == GUI_Mode::Minimizer || ui_shared_state.selected_mode == GUI_Mode::LLG )
    {
        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "n_iterations" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "##llg_n_iterations", &parameters_llg.n_iterations, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_N_Iterations(
                state.get(), parameters_llg.n_iterations, parameters_llg.n_iterations_log );

        ImGui::TextUnformatted( "log every" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "iterations##llg_n_iterations_log", &parameters_llg.n_iterations_log, 0, 0,
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_N_Iterations(
                state.get(), parameters_llg.n_iterations, parameters_llg.n_iterations_log );

        ImGui::Dummy( { 0, 10 } );

        if( ImGui::Checkbox( "##llg_output_any", &parameters_llg.output_any ) )
            Parameters_LLG_Set_Output_General(
                state.get(), parameters_llg.output_any, parameters_llg.output_initial, parameters_llg.output_final );
        ImGui::SameLine();
        if( ImGui::CollapsingHeader( "Output" ) )
        {
            ImGui::Indent( 25 );

            ImGui::TextUnformatted( "folder" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##llg_output_folder", &parameters_llg.output_folder, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_LLG_Set_Output_Folder( state.get(), parameters_llg.output_folder.c_str() );

            ImGui::TextUnformatted( "file tag" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##llg_output_file_tag", &parameters_llg.output_file_tag, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_LLG_Set_Output_Tag( state.get(), parameters_llg.output_file_tag.c_str() );

            if( ImGui::Checkbox( "initial##llg_output_initial", &parameters_llg.output_initial ) )
                Parameters_LLG_Set_Output_General(
                    state.get(), parameters_llg.output_any, parameters_llg.output_initial,
                    parameters_llg.output_final );
            if( ImGui::Checkbox( "final##llg_output_final", &parameters_llg.output_final ) )
                Parameters_LLG_Set_Output_General(
                    state.get(), parameters_llg.output_any, parameters_llg.output_initial,
                    parameters_llg.output_final );

            ImGui::Dummy( { 0, 10 } );

            // TODO
            int output_vf_filetype = IO_Fileformat_OVF_text;

            ImGui::TextUnformatted( "Configuration output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox(
                    "write at every step##llg_output_configuration_step", &parameters_llg.output_configuration_step ) )
                Parameters_LLG_Set_Output_Configuration(
                    state.get(), parameters_llg.output_configuration_step, parameters_llg.output_configuration_archive,
                    parameters_llg.output_vf_filetype );
            if( ImGui::Checkbox(
                    "append to archive at every step##llg_output_configuration_archive",
                    &parameters_llg.output_configuration_archive ) )
                Parameters_LLG_Set_Output_Configuration(
                    state.get(), parameters_llg.output_configuration_step, parameters_llg.output_configuration_archive,
                    parameters_llg.output_vf_filetype );
            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Energy output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox( "write at every step##llg_output_energy_step", &parameters_llg.output_energy_step ) )
                Parameters_LLG_Set_Output_Energy(
                    state.get(), parameters_llg.output_energy_step, parameters_llg.output_energy_archive,
                    parameters_llg.output_energy_spin_resolved, parameters_llg.output_energy_divide_by_nspins,
                    parameters_llg.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "append to archive at every step##llg_output_energy_archive",
                    &parameters_llg.output_energy_archive ) )
                Parameters_LLG_Set_Output_Energy(
                    state.get(), parameters_llg.output_energy_step, parameters_llg.output_energy_archive,
                    parameters_llg.output_energy_spin_resolved, parameters_llg.output_energy_divide_by_nspins,
                    parameters_llg.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "spin-resolved energy files##llg_output_energy_spin_resolved",
                    &parameters_llg.output_energy_spin_resolved ) )
                Parameters_LLG_Set_Output_Energy(
                    state.get(), parameters_llg.output_energy_step, parameters_llg.output_energy_archive,
                    parameters_llg.output_energy_spin_resolved, parameters_llg.output_energy_divide_by_nspins,
                    parameters_llg.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "normalize energies by number of spins##llg_output_energy_divide_by_nspins",
                    &parameters_llg.output_energy_divide_by_nspins ) )
                Parameters_LLG_Set_Output_Energy(
                    state.get(), parameters_llg.output_energy_step, parameters_llg.output_energy_archive,
                    parameters_llg.output_energy_spin_resolved, parameters_llg.output_energy_divide_by_nspins,
                    parameters_llg.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "add readability lines in energy files##llg_output_energy_add_readability_lines",
                    &parameters_llg.output_energy_add_readability_lines ) )
                Parameters_LLG_Set_Output_Energy(
                    state.get(), parameters_llg.output_energy_step, parameters_llg.output_energy_archive,
                    parameters_llg.output_energy_spin_resolved, parameters_llg.output_energy_divide_by_nspins,
                    parameters_llg.output_energy_add_readability_lines );
            ImGui::Indent( -15 );

            ImGui::Indent( -25 );
        }

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Convergence limit" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##llg_convergence_limit", &parameters_llg.force_convergence, 0, 0, "%.3e",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_LLG_Set_Convergence( state.get(), parameters_llg.force_convergence );
        }

        ImGui::TextUnformatted( "dt" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "[ps]##llg_dt", &parameters_llg.dt, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Time_Step( state.get(), parameters_llg.dt );

        ImGui::TextUnformatted( "Damping" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##llg_damping", &parameters_llg.damping, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Damping( state.get(), parameters_llg.damping );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Temperature" );
        ImGui::Indent( 15 );
        ImGui::TextUnformatted( "Base" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "[K]##llg_temperature", &parameters_llg.temperature, 0, 0, "%.5f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Temperature( state.get(), parameters_llg.temperature );
        ImGui::TextUnformatted( "Gradient direction" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 180 );
        if( ImGui::InputFloat3(
                "##llg_temperature_gradient_dir", parameters_llg.temperature_gradient_direction, "%.2f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Temperature_Gradient(
                state.get(), parameters_llg.temperature_gradient_inclination,
                parameters_llg.temperature_gradient_direction );
        ImGui::TextUnformatted( "Gradient inclination" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "[K]##llg_temperature_gradient_inclination", &parameters_llg.temperature_gradient_inclination, 0, 0,
                "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Temperature_Gradient(
                state.get(), parameters_llg.temperature_gradient_inclination,
                parameters_llg.temperature_gradient_direction );
        ImGui::Indent( -15 );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Spin torques" );
        ImGui::Indent( 15 );
        std::vector<const char *> stt_approximation{ "Perpendicular current (STT)", "Gradient (SOT)" };
        int stt_index = int( parameters_llg.stt_use_gradient );
        ImGui::SetNextItemWidth( 220 );
        if( ImGui::Combo( "##approximation", &stt_index, stt_approximation.data(), int( stt_approximation.size() ) ) )
        {
            parameters_llg.stt_use_gradient = bool( stt_index );
            Parameters_LLG_Set_STT(
                state.get(), parameters_llg.stt_use_gradient, parameters_llg.stt_magnitude,
                parameters_llg.stt_polarisation_normal );
        }
        ImGui::TextUnformatted( "Magnitude" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##llg_stt_magnitude", &parameters_llg.stt_magnitude, 0, 0, "%.5f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_STT(
                state.get(), parameters_llg.stt_use_gradient, parameters_llg.stt_magnitude,
                parameters_llg.stt_polarisation_normal );
        ImGui::TextUnformatted( "Polarisation" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 180 );
        if( ImGui::InputFloat3(
                "##llg_stt_polarisation_normal", parameters_llg.stt_polarisation_normal, "%.2f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_STT(
                state.get(), parameters_llg.stt_use_gradient, parameters_llg.stt_magnitude,
                parameters_llg.stt_polarisation_normal );
        ImGui::Indent( -15 );
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::MC )
    {
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.mc_apply_to_all );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "n_iterations" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "##mc_n_iterations", &parameters_mc.n_iterations, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MC_Set_N_Iterations( state.get(), parameters_mc.n_iterations, parameters_mc.n_iterations_log );

        ImGui::TextUnformatted( "log every" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "iterations##mc_n_iterations_log", &parameters_mc.n_iterations_log, 0, 0,
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MC_Set_N_Iterations( state.get(), parameters_mc.n_iterations, parameters_mc.n_iterations_log );

        ImGui::Dummy( { 0, 10 } );

        if( ImGui::Checkbox( "##mc_output_any", &parameters_mc.output_any ) )
            Parameters_MC_Set_Output_General(
                state.get(), parameters_mc.output_any, parameters_mc.output_initial, parameters_mc.output_final );
        ImGui::SameLine();
        if( ImGui::CollapsingHeader( "Output" ) )
        {
            ImGui::Indent( 25 );

            ImGui::TextUnformatted( "folder" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##mc_output_folder", &parameters_mc.output_folder, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_MC_Set_Output_Folder( state.get(), parameters_mc.output_folder.c_str() );

            ImGui::TextUnformatted( "file tag" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##mc_output_file_tag", &parameters_mc.output_file_tag, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_MC_Set_Output_Tag( state.get(), parameters_mc.output_file_tag.c_str() );

            if( ImGui::Checkbox( "initial##mc_output_initial", &parameters_mc.output_initial ) )
                Parameters_MC_Set_Output_General(
                    state.get(), parameters_mc.output_any, parameters_mc.output_initial, parameters_mc.output_final );
            if( ImGui::Checkbox( "final##mc_output_final", &parameters_mc.output_final ) )
                Parameters_MC_Set_Output_General(
                    state.get(), parameters_mc.output_any, parameters_mc.output_initial, parameters_mc.output_final );

            ImGui::Dummy( { 0, 10 } );

            // TODO
            int output_vf_filetype = IO_Fileformat_OVF_text;

            ImGui::TextUnformatted( "Configuration output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox(
                    "write at every step##mc_output_configuration_step", &parameters_mc.output_configuration_step ) )
                Parameters_MC_Set_Output_Configuration(
                    state.get(), parameters_mc.output_configuration_step, parameters_mc.output_configuration_archive,
                    parameters_mc.output_vf_filetype );
            if( ImGui::Checkbox(
                    "append to archive at every step##mc_output_configuration_archive",
                    &parameters_mc.output_configuration_archive ) )
                Parameters_MC_Set_Output_Configuration(
                    state.get(), parameters_mc.output_configuration_step, parameters_mc.output_configuration_archive,
                    parameters_mc.output_vf_filetype );
            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Energy output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox( "write at every step##mc_output_energy_step", &parameters_mc.output_energy_step ) )
                Parameters_MC_Set_Output_Energy(
                    state.get(), parameters_mc.output_energy_step, parameters_mc.output_energy_archive,
                    parameters_mc.output_energy_spin_resolved, parameters_mc.output_energy_divide_by_nspins,
                    parameters_mc.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "append to archive at every step##mc_output_energy_archive",
                    &parameters_mc.output_energy_archive ) )
                Parameters_MC_Set_Output_Energy(
                    state.get(), parameters_mc.output_energy_step, parameters_mc.output_energy_archive,
                    parameters_mc.output_energy_spin_resolved, parameters_mc.output_energy_divide_by_nspins,
                    parameters_mc.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "spin-resolved energy files##mc_output_energy_spin_resolved",
                    &parameters_mc.output_energy_spin_resolved ) )
                Parameters_MC_Set_Output_Energy(
                    state.get(), parameters_mc.output_energy_step, parameters_mc.output_energy_archive,
                    parameters_mc.output_energy_spin_resolved, parameters_mc.output_energy_divide_by_nspins,
                    parameters_mc.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "normalize energies by number of spins##mc_output_energy_divide_by_nspins",
                    &parameters_mc.output_energy_divide_by_nspins ) )
                Parameters_MC_Set_Output_Energy(
                    state.get(), parameters_mc.output_energy_step, parameters_mc.output_energy_archive,
                    parameters_mc.output_energy_spin_resolved, parameters_mc.output_energy_divide_by_nspins,
                    parameters_mc.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "add readability lines in energy files##mc_output_energy_add_readability_lines",
                    &parameters_mc.output_energy_add_readability_lines ) )
                Parameters_MC_Set_Output_Energy(
                    state.get(), parameters_mc.output_energy_step, parameters_mc.output_energy_archive,
                    parameters_mc.output_energy_spin_resolved, parameters_mc.output_energy_divide_by_nspins,
                    parameters_mc.output_energy_add_readability_lines );
            ImGui::Indent( -15 );

            ImGui::Indent( -25 );
        }

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Temperature" );
        ImGui::Indent( 15 );
        ImGui::TextUnformatted( "Base" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "[K]##mc_temperature", &parameters_mc.temperature, 0, 0, "%.5f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MC_Set_Temperature( state.get(), parameters_mc.temperature );
        ImGui::Indent( -15 );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Metropolis algorithm" );
        ImGui::Indent( 15 );
        if( ImGui::Checkbox( "randomly sample spin", &parameters_mc.metropolis_random_sample ) )
            Parameters_MC_Set_Random_Sample( state.get(), parameters_mc.metropolis_random_sample );
        if( ImGui::Checkbox( "restrict sampling to a cone", &parameters_mc.metropolis_step_cone ) )
            Parameters_MC_Set_Metropolis_Cone(
                state.get(), parameters_mc.metropolis_step_cone, parameters_mc.metropolis_cone_angle,
                parameters_mc.metropolis_cone_adaptive, parameters_mc.acceptance_ratio_target );
        if( ImGui::Checkbox( "dynamically adapt the cone radius", &parameters_mc.metropolis_cone_adaptive ) )
            Parameters_MC_Set_Metropolis_Cone(
                state.get(), parameters_mc.metropolis_step_cone, parameters_mc.metropolis_cone_angle,
                parameters_mc.metropolis_cone_adaptive, parameters_mc.acceptance_ratio_target );
        ImGui::TextUnformatted( "cone angle" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 100 );
        if( ImGui::InputFloat(
                "[deg]##mc_cone_angle", &parameters_mc.metropolis_cone_angle, 0, 0, "%.5f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MC_Set_Metropolis_Cone(
                state.get(), parameters_mc.metropolis_step_cone, parameters_mc.metropolis_cone_angle,
                parameters_mc.metropolis_cone_adaptive, parameters_mc.acceptance_ratio_target );
        ImGui::TextUnformatted( "target acceptance ratio" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 100 );
        if( ImGui::InputFloat(
                "##mc_acceptance_ratio_target", &parameters_mc.acceptance_ratio_target, 0, 0, "%.5f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MC_Set_Metropolis_Cone(
                state.get(), parameters_mc.metropolis_step_cone, parameters_mc.metropolis_cone_angle,
                parameters_mc.metropolis_cone_adaptive, parameters_mc.acceptance_ratio_target );
        ImGui::Indent( -15 );
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::GNEB )
    {
        ImGui::TextUnformatted( "n_iterations" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "##gneb_n_iterations", &parameters_gneb.n_iterations, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_GNEB_Set_N_Iterations(
                state.get(), parameters_gneb.n_iterations, parameters_gneb.n_iterations_log );

        ImGui::TextUnformatted( "log every" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "iterations##gneb_n_iterations_log", &parameters_gneb.n_iterations_log, 0, 0,
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_GNEB_Set_N_Iterations(
                state.get(), parameters_gneb.n_iterations, parameters_gneb.n_iterations_log );

        ImGui::Dummy( { 0, 10 } );

        if( ImGui::Checkbox( "##gneb_output_any", &parameters_gneb.output_any ) )
            Parameters_GNEB_Set_Output_General(
                state.get(), parameters_gneb.output_any, parameters_gneb.output_initial, parameters_gneb.output_final );
        ImGui::SameLine();
        if( ImGui::CollapsingHeader( "Output" ) )
        {
            ImGui::Indent( 25 );

            ImGui::TextUnformatted( "folder" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##gneb_output_folder", &parameters_gneb.output_folder, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_GNEB_Set_Output_Folder( state.get(), parameters_gneb.output_folder.c_str() );

            ImGui::TextUnformatted( "file tag" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##gneb_output_file_tag", &parameters_gneb.output_file_tag, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_GNEB_Set_Output_Tag( state.get(), parameters_gneb.output_file_tag.c_str() );

            if( ImGui::Checkbox( "initial##gneb_output_initial", &parameters_gneb.output_initial ) )
                Parameters_GNEB_Set_Output_General(
                    state.get(), parameters_gneb.output_any, parameters_gneb.output_initial,
                    parameters_gneb.output_final );
            if( ImGui::Checkbox( "final##gneb_output_final", &parameters_gneb.output_final ) )
                Parameters_GNEB_Set_Output_General(
                    state.get(), parameters_gneb.output_any, parameters_gneb.output_initial,
                    parameters_gneb.output_final );

            ImGui::Dummy( { 0, 10 } );

            // TODO
            int output_vf_filetype = IO_Fileformat_OVF_text;

            ImGui::TextUnformatted( "Chain output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox( "write at every step##gneb_output_chain_step", &parameters_gneb.output_chain_step ) )
                Parameters_GNEB_Set_Output_Chain(
                    state.get(), parameters_gneb.output_chain_step, parameters_gneb.output_vf_filetype );
            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Energy output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox(
                    "write at every step##gneb_output_energies_step", &parameters_gneb.output_energies_step ) )
                Parameters_GNEB_Set_Output_Energies(
                    state.get(), parameters_gneb.output_energies_step, parameters_gneb.output_energies_interpolated,
                    parameters_gneb.output_energies_divide_by_nspins,
                    parameters_gneb.output_energies_add_readability_lines );
            if( ImGui::Checkbox(
                    "normalize energies by number of spins##gneb_output_energies_interpolated",
                    &parameters_gneb.output_energies_interpolated ) )
                Parameters_GNEB_Set_Output_Energies(
                    state.get(), parameters_gneb.output_energies_step, parameters_gneb.output_energies_interpolated,
                    parameters_gneb.output_energies_divide_by_nspins,
                    parameters_gneb.output_energies_add_readability_lines );
            if( ImGui::Checkbox(
                    "normalize energies by number of spins##gneb_output_energies_divide_by_nspins",
                    &parameters_gneb.output_energies_divide_by_nspins ) )
                Parameters_GNEB_Set_Output_Energies(
                    state.get(), parameters_gneb.output_energies_step, parameters_gneb.output_energies_interpolated,
                    parameters_gneb.output_energies_divide_by_nspins,
                    parameters_gneb.output_energies_add_readability_lines );
            if( ImGui::Checkbox(
                    "add readability lines in energy files##gneb_output_energies_add_readability_lines",
                    &parameters_gneb.output_energies_add_readability_lines ) )
                Parameters_GNEB_Set_Output_Energies(
                    state.get(), parameters_gneb.output_energies_step, parameters_gneb.output_energies_interpolated,
                    parameters_gneb.output_energies_divide_by_nspins,
                    parameters_gneb.output_energies_add_readability_lines );
            ImGui::Indent( -15 );

            ImGui::Indent( -25 );
        }

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Convergence limit" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##gneb_convergence_limit", &parameters_gneb.force_convergence, 0, 0, "%.3e",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Convergence( state.get(), parameters_gneb.force_convergence );
        }

        std::vector<const char *> gneb_image_types{
            "Normal",
            "Climbing",
            "Falling",
            "Stationary",
        };
        ImGui::TextUnformatted( "Image type" );
        ImGui::Indent( 15 );
        if( ImGui::ListBox( "##gneb_image_type", &gneb_image_type, gneb_image_types.data(), 4 ) )
        {
            Parameters_GNEB_Set_Climbing_Falling( state.get(), gneb_image_type );
        }
        ImGui::Indent( -15 );

        ImGui::TextUnformatted( "Spring constant" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##gneb_spring_constant", &parameters_gneb.spring_constant, 0, 0, "%.3e",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            for( int idx = 0; idx < Chain_Get_NOI( state.get() ); ++idx )
                Parameters_GNEB_Set_Spring_Constant( state.get(), parameters_gneb.spring_constant, idx );
        }

        ImGui::TextUnformatted( "Spring/Energy ratio" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##gneb_spring_energy_ratio", &parameters_gneb.spring_force_ratio, 0, 0, "%.2f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Spring_Force_Ratio( state.get(), parameters_gneb.spring_force_ratio );
        }

        ImGui::TextUnformatted( "Path shortening constant" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##gneb_path_shortening", &parameters_gneb.path_shortening_constant, 0, 0, "%.3e",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Path_Shortening_Constant( state.get(), parameters_gneb.path_shortening_constant );
        }
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::MMF )
    {
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.mmf_apply_to_all );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "n_iterations" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "##mmf_n_iterations", &parameters_mmf.n_iterations, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MMF_Set_N_Iterations(
                state.get(), parameters_mmf.n_iterations, parameters_mmf.n_iterations_log );

        ImGui::TextUnformatted( "log every" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputInt(
                "iterations##mmf_n_iterations_log", &parameters_mmf.n_iterations_log, 0, 0,
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_MMF_Set_N_Iterations(
                state.get(), parameters_mmf.n_iterations, parameters_mmf.n_iterations_log );

        ImGui::Dummy( { 0, 10 } );

        if( ImGui::Checkbox( "##mmf_output_any", &parameters_mmf.output_any ) )
            Parameters_MMF_Set_Output_General(
                state.get(), parameters_mmf.output_any, parameters_mmf.output_initial, parameters_mmf.output_final );
        ImGui::SameLine();
        if( ImGui::CollapsingHeader( "Output" ) )
        {
            ImGui::Indent( 25 );

            ImGui::TextUnformatted( "folder" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##mmf_output_folder", &parameters_mmf.output_folder, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_MMF_Set_Output_Folder( state.get(), parameters_mmf.output_folder.c_str() );

            ImGui::TextUnformatted( "file tag" );
            ImGui::SameLine();
            ImGui::SetNextItemWidth( 80 );
            if( ImGui::InputText(
                    "##mmf_output_file_tag", &parameters_mmf.output_file_tag, ImGuiInputTextFlags_EnterReturnsTrue ) )
                Parameters_MMF_Set_Output_Tag( state.get(), parameters_mmf.output_file_tag.c_str() );

            if( ImGui::Checkbox( "initial##mmf_output_initial", &parameters_mmf.output_initial ) )
                Parameters_MMF_Set_Output_General(
                    state.get(), parameters_mmf.output_any, parameters_mmf.output_initial,
                    parameters_mmf.output_final );
            if( ImGui::Checkbox( "final##mmf_output_final", &parameters_mmf.output_final ) )
                Parameters_MMF_Set_Output_General(
                    state.get(), parameters_mmf.output_any, parameters_mmf.output_initial,
                    parameters_mmf.output_final );

            ImGui::Dummy( { 0, 10 } );

            // TODO
            int output_vf_filetype = IO_Fileformat_OVF_text;

            ImGui::TextUnformatted( "Configuration output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox(
                    "write at every step##mmf_output_configuration_step", &parameters_mmf.output_configuration_step ) )
                Parameters_MMF_Set_Output_Configuration(
                    state.get(), parameters_mmf.output_configuration_step, parameters_mmf.output_configuration_archive,
                    parameters_mmf.output_vf_filetype );
            if( ImGui::Checkbox(
                    "append to archive at every step##mmf_output_configuration_archive",
                    &parameters_mmf.output_configuration_archive ) )
                Parameters_MMF_Set_Output_Configuration(
                    state.get(), parameters_mmf.output_configuration_step, parameters_mmf.output_configuration_archive,
                    parameters_mmf.output_vf_filetype );
            ImGui::Indent( -15 );

            ImGui::Dummy( { 0, 10 } );

            ImGui::TextUnformatted( "Energy output" );
            ImGui::Indent( 15 );
            if( ImGui::Checkbox( "write at every step##mmf_output_energy_step", &parameters_mmf.output_energy_step ) )
                Parameters_MMF_Set_Output_Energy(
                    state.get(), parameters_mmf.output_energy_step, parameters_mmf.output_energy_archive,
                    parameters_mmf.output_energy_spin_resolved, parameters_mmf.output_energy_divide_by_nspins,
                    parameters_mmf.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "append to archive at every step##mmf_output_energy_archive",
                    &parameters_mmf.output_energy_archive ) )
                Parameters_MMF_Set_Output_Energy(
                    state.get(), parameters_mmf.output_energy_step, parameters_mmf.output_energy_archive,
                    parameters_mmf.output_energy_spin_resolved, parameters_mmf.output_energy_divide_by_nspins,
                    parameters_mmf.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "spin-resolved energy files##mmf_output_energy_spin_resolved",
                    &parameters_mmf.output_energy_spin_resolved ) )
                Parameters_MMF_Set_Output_Energy(
                    state.get(), parameters_mmf.output_energy_step, parameters_mmf.output_energy_archive,
                    parameters_mmf.output_energy_spin_resolved, parameters_mmf.output_energy_divide_by_nspins,
                    parameters_mmf.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "normalize energies by number of spins##mmf_output_energy_divide_by_nspins",
                    &parameters_mmf.output_energy_divide_by_nspins ) )
                Parameters_MMF_Set_Output_Energy(
                    state.get(), parameters_mmf.output_energy_step, parameters_mmf.output_energy_archive,
                    parameters_mmf.output_energy_spin_resolved, parameters_mmf.output_energy_divide_by_nspins,
                    parameters_mmf.output_energy_add_readability_lines );
            if( ImGui::Checkbox(
                    "add readability lines in energy files##mmf_output_energy_add_readability_lines",
                    &parameters_mmf.output_energy_add_readability_lines ) )
                Parameters_MMF_Set_Output_Energy(
                    state.get(), parameters_mmf.output_energy_step, parameters_mmf.output_energy_archive,
                    parameters_mmf.output_energy_spin_resolved, parameters_mmf.output_energy_divide_by_nspins,
                    parameters_mmf.output_energy_add_readability_lines );
            ImGui::Indent( -15 );

            ImGui::Indent( -25 );
        }
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::EMA )
    {
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.ema_apply_to_all );

        ImGui::Dummy( { 0, 10 } );

        ImGui::TextUnformatted( "Number of modes" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 100 );
        if( ImGui::InputInt( "##ema_n_modes", &parameters_ema.n_modes, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_EMA_Set_N_Modes( state.get(), parameters_ema.n_modes );

        ImGui::TextUnformatted( "Follow mode" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 100 );
        if( ImGui::InputInt(
                "##ema_n_mode_follow", &parameters_ema.n_mode_follow, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_EMA_Set_N_Mode_Follow( state.get(), parameters_ema.n_mode_follow );

        ImGui::TextUnformatted( "Frequency" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##ema_frequency", &parameters_ema.frequency, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_EMA_Set_Frequency( state.get(), parameters_ema.frequency );

        ImGui::TextUnformatted( "Amplitude" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 80 );
        if( ImGui::InputFloat(
                "##ema_amplitude", &parameters_ema.amplitude, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_EMA_Set_Amplitude( state.get(), parameters_ema.amplitude );

        if( ImGui::Checkbox( "snapshot mode", &parameters_ema.snapshot ) )
            Parameters_EMA_Set_Snapshot( state.get(), parameters_ema.snapshot );
    }
}

void ParametersWidget::update_data()
{
    // ----------------- MC
    Parameters_MC_Get_N_Iterations( state.get(), &parameters_mc.n_iterations, &parameters_mc.n_iterations_log );
    parameters_mc.temperature = Parameters_LLG_Get_Temperature( state.get() );
    // Output
    parameters_mc.output_folder   = Parameters_MC_Get_Output_Folder( state.get() );
    parameters_mc.output_file_tag = Parameters_MC_Get_Output_Tag( state.get() );
    Parameters_MC_Get_Output_General(
        state.get(), &parameters_mc.output_any, &parameters_mc.output_initial, &parameters_mc.output_final );
    Parameters_MC_Get_Output_Configuration(
        state.get(), &parameters_mc.output_configuration_step, &parameters_mc.output_configuration_archive,
        &parameters_mc.output_vf_filetype );
    Parameters_MC_Get_Output_Energy(
        state.get(), &parameters_mc.output_energy_step, &parameters_mc.output_energy_archive,
        &parameters_mc.output_energy_spin_resolved, &parameters_mc.output_energy_divide_by_nspins,
        &parameters_mc.output_energy_add_readability_lines );

    // ----------------- LLG
    Parameters_LLG_Get_N_Iterations( state.get(), &parameters_llg.n_iterations, &parameters_llg.n_iterations_log );
    parameters_llg.force_convergence = Parameters_LLG_Get_Convergence( state.get() );
    parameters_llg.dt                = Parameters_LLG_Get_Time_Step( state.get() );
    parameters_llg.damping           = Parameters_LLG_Get_Damping( state.get() );
    parameters_llg.temperature       = Parameters_LLG_Get_Temperature( state.get() );
    Parameters_LLG_Get_Temperature_Gradient(
        state.get(), parameters_llg.temperature_gradient_direction, &parameters_llg.temperature_gradient_inclination );
    Parameters_LLG_Get_STT(
        state.get(), &parameters_llg.stt_use_gradient, &parameters_llg.stt_magnitude,
        parameters_llg.stt_polarisation_normal );
    parameters_llg.direct_minimization = Parameters_LLG_Get_Direct_Minimization( state.get() );
    // Output
    parameters_llg.output_folder   = Parameters_LLG_Get_Output_Folder( state.get() );
    parameters_llg.output_file_tag = Parameters_LLG_Get_Output_Tag( state.get() );
    Parameters_LLG_Get_Output_General(
        state.get(), &parameters_llg.output_any, &parameters_llg.output_initial, &parameters_llg.output_final );
    Parameters_LLG_Get_Output_Configuration(
        state.get(), &parameters_llg.output_configuration_step, &parameters_llg.output_configuration_archive,
        &parameters_llg.output_vf_filetype );
    Parameters_LLG_Get_Output_Energy(
        state.get(), &parameters_llg.output_energy_step, &parameters_llg.output_energy_archive,
        &parameters_llg.output_energy_spin_resolved, &parameters_llg.output_energy_divide_by_nspins,
        &parameters_llg.output_energy_add_readability_lines );

    // ----------------- GNEB
    Parameters_GNEB_Get_N_Iterations( state.get(), &parameters_gneb.n_iterations, &parameters_gneb.n_iterations_log );
    parameters_gneb.force_convergence        = Parameters_GNEB_Get_Convergence( state.get() );
    gneb_image_type                          = Parameters_GNEB_Get_Climbing_Falling( state.get() );
    parameters_gneb.spring_constant          = Parameters_GNEB_Get_Spring_Constant( state.get() );
    parameters_gneb.spring_force_ratio       = Parameters_GNEB_Get_Spring_Force_Ratio( state.get() );
    parameters_gneb.path_shortening_constant = Parameters_GNEB_Get_Path_Shortening_Constant( state.get() );
    // Output
    parameters_gneb.output_folder   = Parameters_GNEB_Get_Output_Folder( state.get() );
    parameters_gneb.output_file_tag = Parameters_GNEB_Get_Output_Tag( state.get() );
    Parameters_GNEB_Get_Output_General(
        state.get(), &parameters_gneb.output_any, &parameters_gneb.output_initial, &parameters_gneb.output_final );
    Parameters_GNEB_Get_Output_Chain(
        state.get(), &parameters_gneb.output_chain_step, &parameters_gneb.output_vf_filetype );
    Parameters_GNEB_Get_Output_Energies(
        state.get(), &parameters_gneb.output_energies_step, &parameters_gneb.output_energies_interpolated,
        &parameters_gneb.output_energies_divide_by_nspins, &parameters_gneb.output_energies_add_readability_lines );

    // ----------------- MMF
    Parameters_MMF_Get_N_Iterations( state.get(), &parameters_mmf.n_iterations, &parameters_mmf.n_iterations_log );
    // Output
    parameters_mmf.output_folder   = Parameters_MMF_Get_Output_Folder( state.get() );
    parameters_mmf.output_file_tag = Parameters_MMF_Get_Output_Tag( state.get() );
    Parameters_MMF_Get_Output_General(
        state.get(), &parameters_mmf.output_any, &parameters_mmf.output_initial, &parameters_mmf.output_final );
    Parameters_MMF_Get_Output_Configuration(
        state.get(), &parameters_mmf.output_configuration_step, &parameters_mmf.output_configuration_archive,
        &parameters_mmf.output_vf_filetype );
    Parameters_MMF_Get_Output_Energy(
        state.get(), &parameters_mmf.output_energy_step, &parameters_mmf.output_energy_archive,
        &parameters_mmf.output_energy_spin_resolved, &parameters_mmf.output_energy_divide_by_nspins,
        &parameters_mmf.output_energy_add_readability_lines );

    // ----------------- EMA
    parameters_ema.amplitude     = Parameters_EMA_Get_Amplitude( state.get() );
    parameters_ema.frequency     = Parameters_EMA_Get_Frequency( state.get() );
    parameters_ema.snapshot      = Parameters_EMA_Get_Snapshot( state.get() );
    parameters_ema.n_modes       = Parameters_EMA_Get_N_Modes( state.get() );
    parameters_ema.n_mode_follow = Parameters_EMA_Get_N_Mode_Follow( state.get() );
}

} // namespace ui