#include <parameters_widget.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>

#include <imgui/imgui.h>

namespace ui
{

ParametersWidget::ParametersWidget( bool & show, std::shared_ptr<State> state, GUI_Mode & selected_mode )
        : show_( show ), state( state ), selected_mode( selected_mode )
{
    llg_convergence = Parameters_LLG_Get_Convergence( state.get() );
    llg_dt          = Parameters_LLG_Get_Time_Step( state.get() );

    gneb_convergence         = Parameters_GNEB_Get_Convergence( state.get() );
    gneb_image_type          = Parameters_GNEB_Get_Climbing_Falling( state.get() );
    gneb_spring_constant     = Parameters_GNEB_Get_Spring_Constant( state.get() );
    gneb_spring_energy_ratio = Parameters_GNEB_Get_Spring_Force_Ratio( state.get() );
    gneb_path_shortening     = Parameters_GNEB_Get_Path_Shortening_Constant( state.get() );
}

void ParametersWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Parameters", &show_ );

    if( selected_mode == GUI_Mode::Minimizer )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }

        ImGui::TextUnformatted( "Convergence limit" );
        ImGui::SameLine();
        if( ImGui::InputFloat(
                "##llg_convergence_limit", &llg_convergence, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_LLG_Set_Convergence( state.get(), llg_convergence );
        }

        ImGui::TextUnformatted( "dt" );
        ImGui::SameLine();
        if( ImGui::InputFloat( "[ps]##min", &llg_dt, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Time_Step( state.get(), llg_dt );
    }
    else if( selected_mode == GUI_Mode::MC )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::LLG )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }

        ImGui::TextUnformatted( "dt" );
        ImGui::SameLine();
        if( ImGui::InputFloat( "[ps]##llg", &llg_dt, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
            Parameters_LLG_Set_Time_Step( state.get(), llg_dt );
    }
    else if( selected_mode == GUI_Mode::GNEB )
    {
        ImGui::TextUnformatted( "Convergence limit" );
        ImGui::SameLine();
        if( ImGui::InputFloat(
                "##gneb_convergence_limit", &gneb_convergence, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Convergence( state.get(), gneb_convergence );
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
        if( ImGui::InputFloat(
                "##gneb_spring_constant", &gneb_spring_constant, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            for( int idx = 0; idx < Chain_Get_NOI( state.get() ); ++idx )
                Parameters_GNEB_Set_Spring_Constant( state.get(), gneb_spring_constant, idx );
        }

        ImGui::TextUnformatted( "Spring/Energy ratio" );
        ImGui::SameLine();
        if( ImGui::InputFloat(
                "##gneb_spring_energy_ratio", &gneb_spring_energy_ratio, 0, 0, "%.2f",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Spring_Force_Ratio( state.get(), gneb_spring_energy_ratio );
        }

        ImGui::TextUnformatted( "Path shortening constant" );
        ImGui::SameLine();
        if( ImGui::InputFloat(
                "##gneb_path_shortening", &gneb_path_shortening, 0, 0, "%.3e", ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Parameters_GNEB_Set_Path_Shortening_Constant( state.get(), gneb_path_shortening );
        }
    }
    else if( selected_mode == GUI_Mode::MMF )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }
    else if( selected_mode == GUI_Mode::EMA )
    {
        if( ImGui::Button( "Apply to all images" ) )
        {
        }
    }

    ImGui::End();
}

void ParametersWidget::update_data() {}

} // namespace ui