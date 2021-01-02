#include <configurations_widget.hpp>
#include <widgets.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>

#include <imgui/imgui.h>

namespace ui
{

ConfigurationsWidget::ConfigurationsWidget(
    bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer )
        : show_( show ), state( state ), rendering_layer( rendering_layer )
{
}

void ConfigurationsWidget::reset_settings()
{
    for( int dim = 0; dim < 3; ++dim )
    {
        this->pos[dim]         = 0;
        this->border_rect[dim] = -1, -1, -1;
    }
    this->border_cyl = -1;
    this->border_sph = -1;
    this->inverted   = false;

    this->temperature = 0;

    this->sk_radius  = 15;
    this->sk_speed   = 1;
    this->sk_phase   = 0;
    this->sk_up_down = false;
    this->sk_achiral = false;
    this->sk_rl      = false;

    this->hopfion_radius = 10;
    this->hopfion_order  = 1;

    this->spiral_angle    = 0;
    this->spiral_axis[0]  = 0;
    this->spiral_axis[1]  = 0;
    this->spiral_axis[2]  = 1;
    this->spiral_qmag     = 1;
    this->spiral_qvec[0]  = 1;
    this->spiral_qvec[1]  = 0;
    this->spiral_qvec[2]  = 0;
    this->spiral_q2       = false;
    this->spiral_qmag2    = 1;
    this->spiral_qvec2[0] = 1;
    this->spiral_qvec2[1] = 0;
    this->spiral_qvec2[2] = 0;
}

void ConfigurationsWidget::show()
{
    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 300, 300 }, { 800, 999999 } );

    ImGui::Begin( "Configurations", &show_ );

    if( ImGui::CollapsingHeader( "Settings" ) )
    {
        ImGui::Indent( 15 );

        if( ImGui::Button( "Reset" ) )
            this->reset_settings();

        ImGui::Dummy( { 0, 5 } );

        ImGui::TextUnformatted( "pos" );
        ImGui::SameLine();
        ImGui::InputFloat3( "##configurations_pos", pos );

        ImGui::TextUnformatted( "Rectangular border" );
        ImGui::SameLine();
        ImGui::InputFloat3( "##configurations_border_rect", border_rect );

        ImGui::TextUnformatted( "Cylindrical border" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 50 );
        ImGui::InputFloat( "##configurations_border_cyl", &border_cyl );

        ImGui::TextUnformatted( "Spherical border" );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 50 );
        ImGui::InputFloat( "##configurations_border_sph", &border_sph );

        ImGui::TextUnformatted( "invert" );
        ImGui::SameLine();
        ImGui::Checkbox( "##configurations_inverted", &inverted );

        ImGui::Indent( -15 );
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Random" ) )
    {
        Configuration_Random( state.get(), pos, border_rect, border_cyl, border_sph, inverted );
        rendering_layer.needs_data();
    }
    ImGui::SameLine();
    if( ImGui::Button( "+z" ) )
    {
        Configuration_PlusZ( state.get(), pos, border_rect, border_cyl, border_sph, inverted );
        rendering_layer.needs_data();
    }
    ImGui::SameLine();
    if( ImGui::Button( "-z" ) )
    {
        Configuration_MinusZ( state.get(), pos, border_rect, border_cyl, border_sph, inverted );
        rendering_layer.needs_data();
    }

    ImGui::Dummy( { 0, 10 } );

    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_noise", &temperature );
    ImGui::SameLine();
    if( ImGui::Button( "Add noise" ) )
    {
        Configuration_Add_Noise_Temperature(
            state.get(), temperature, pos, border_rect, border_cyl, border_sph, inverted );

        Chain_Update_Data( state.get() );
        rendering_layer.needs_data();
    }

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Skyrmion" ) )
    {
        Configuration_Skyrmion(
            state.get(), sk_radius, sk_speed, sk_phase, sk_up_down, sk_achiral, sk_rl, pos, border_rect, border_cyl,
            border_sph, inverted );

        rendering_layer.needs_data();
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "Radius" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_radius", &sk_radius );
    ImGui::TextUnformatted( "Speed" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_speed", &sk_speed );
    ImGui::TextUnformatted( "Phase" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_sk_phase", &sk_phase );

    ImGui::TextUnformatted( "Up" );
    ImGui::SameLine();
    widgets::toggle_button( "##configurations_sk_up_down", &sk_up_down, false );
    ImGui::SameLine();
    ImGui::TextUnformatted( "Down" );

    ImGui::TextUnformatted( "Achiral" );
    ImGui::SameLine();
    ImGui::Checkbox( "##configurations_sk_achiral", &sk_achiral );

    ImGui::TextUnformatted( "Invert vorticity" );
    ImGui::SameLine();
    ImGui::Checkbox( "##configurations_sk_rl", &sk_rl );

    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    if( ImGui::Button( "Hopfion" ) )
    {
        Configuration_Hopfion(
            state.get(), hopfion_radius, hopfion_order, pos, border_rect, border_cyl, border_sph, inverted );

        rendering_layer.needs_data();
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "Radius" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 60 );
    ImGui::InputFloat( "##configurations_hopfion_radius", &hopfion_radius );

    ImGui::TextUnformatted( "Order" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 80 );
    ImGui::InputInt( "##configurations_hopfion_order", &hopfion_order );

    ImGui::Indent( -15 );

    ImGui::Dummy( { 0, 10 } );
    ImGui::Separator();
    ImGui::Dummy( { 0, 10 } );

    const char * direction_type = "Real Lattice";
    // if( comboBox_SS->currentText() == "Real Lattice" )
    //     direction_type = "Real Lattice";
    // else if( comboBox_SS->currentText() == "Reciprocal Lattice" )
    //     direction_type = "Reciprocal Lattice";
    // else if( comboBox_SS->currentText() == "Real Space" )
    //     direction_type = "Real Space";

    if( ImGui::Button( "Spiral" ) )
    {
        // Normalize
        float absq = std::sqrt(
            spiral_qvec[0] * spiral_qvec[0] + spiral_qvec[1] * spiral_qvec[1] + spiral_qvec[2] * spiral_qvec[2] );
        if( absq == 0 )
        {
            spiral_qvec[0] = 0;
            spiral_qvec[1] = 0;
            spiral_qvec[2] = 1;
        }

        // Scale
        for( int dim = 0; dim < 3; ++dim )
            spiral_qvec[dim] *= spiral_qmag;

        // Create Spin Spiral
        if( spiral_q2 )
        {
            // Normalize
            float absq2 = std::sqrt(
                spiral_qvec2[0] * spiral_qvec2[0] + spiral_qvec2[1] * spiral_qvec2[1]
                + spiral_qvec2[2] * spiral_qvec2[2] );
            if( absq == 0 )
            {
                spiral_qvec2[0] = 0;
                spiral_qvec2[1] = 0;
                spiral_qvec2[2] = 1;
            }

            // Scale
            for( int dim = 0; dim < 3; ++dim )
                spiral_qvec2[dim] *= spiral_qmag2;

            Configuration_SpinSpiral_2q(
                state.get(), direction_type, spiral_qvec, spiral_qvec2, spiral_axis, spiral_angle, pos, border_rect,
                border_cyl, border_sph, inverted );
        }
        else
            Configuration_SpinSpiral(
                state.get(), direction_type, spiral_qvec, spiral_axis, spiral_angle, pos, border_rect, border_cyl,
                border_sph, inverted );
    }

    ImGui::Indent( 15 );

    ImGui::TextUnformatted( "q" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_spiral_qmag", &spiral_qmag );

    ImGui::TextUnformatted( "q dir" );
    ImGui::SameLine();
    ImGui::InputFloat3( "##configurations_spiral_qvec", spiral_qvec );

    ImGui::TextUnformatted( "axis" );
    ImGui::SameLine();
    ImGui::InputFloat3( "##configurations_spiral_axis", spiral_axis );

    ImGui::TextUnformatted( "angle" );
    ImGui::SameLine();
    ImGui::SetNextItemWidth( 50 );
    ImGui::InputFloat( "##configurations_spiral_angle", &spiral_angle );

    ImGui::Indent( -15 );

    ImGui::End();
}

void ConfigurationsWidget::update_data() {}

} // namespace ui