#include <parameters_widget.hpp>

#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>

#include <imgui/imgui.h>

namespace ui
{

ParametersWidget::ParametersWidget( bool & show, std::shared_ptr<State> state, GUI_Mode & selected_mode )
        : show_( show ), state( state ), selected_mode( selected_mode )
{
    dt = Parameters_LLG_Get_Time_Step( state.get() );
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

        if( ImGui::InputFloat( "dt [ps]", &dt ) )
            Parameters_LLG_Set_Time_Step( state.get(), dt );
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
    }
    else if( selected_mode == GUI_Mode::GNEB )
    {
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