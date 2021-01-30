#include <parameters_widget.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>

#include <imgui/imgui.h>
#include <imgui/misc/cpp/imgui_stdlib.h>

namespace ui
{

ParametersWidget::ParametersWidget( bool & show, std::shared_ptr<State> state, UiSharedState & ui_shared_state )
        : show_( show ), state( state ), ui_shared_state( ui_shared_state )
{
    update_data();
}

void ParametersWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Parameters", &show_ );

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
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::GNEB )
    {
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
    }
    else if( ui_shared_state.selected_mode == GUI_Mode::EMA )
    {
        ImGui::Checkbox( "Apply to all images", &ui_shared_state.ema_apply_to_all );
    }

    ImGui::End();
}

void ParametersWidget::update_data()
{
    parameters_llg.force_convergence = Parameters_LLG_Get_Convergence( state.get() );
    parameters_llg.dt                = Parameters_LLG_Get_Time_Step( state.get() );

    parameters_gneb.force_convergence        = Parameters_GNEB_Get_Convergence( state.get() );
    gneb_image_type                          = Parameters_GNEB_Get_Climbing_Falling( state.get() );
    parameters_gneb.spring_constant          = Parameters_GNEB_Get_Spring_Constant( state.get() );
    parameters_gneb.spring_force_ratio       = Parameters_GNEB_Get_Spring_Force_Ratio( state.get() );
    parameters_gneb.path_shortening_constant = Parameters_GNEB_Get_Path_Shortening_Constant( state.get() );
}

} // namespace ui