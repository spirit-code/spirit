#include <plots_widget.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <imgui/imgui.h>

#include <implot/implot.h>

#include <fmt/format.h>

#include <string>
#include <vector>

namespace ui
{

PlotsWidget::PlotsWidget( bool & show, std::shared_ptr<State> state ) : show_( show ), state( state ) {}

void PlotsWidget::show()
{
    auto & style = ImGui::GetStyle();

    static bool plot_image_energies        = true;
    static bool plot_interpolated_energies = true;

    static int history_size = 200;
    static std::vector<float> force_history( history_size );
    static std::vector<float> iteration_history( history_size );
    static int force_index = 0;
    static float force_min = 0;
    static float force_max = 0;

    static bool animate = true;

    static int n_interpolate = Parameters_GNEB_Get_N_Energy_Interpolations( state.get() );
    static std::vector<float> rx( 1, 0 );
    static std::vector<float> energies( 1, 0 );
    static std::vector<float> rx_interpolated( 1, 0 );
    static std::vector<float> energies_interpolated( 1, 0 );

    if( Simulation_Running_Anywhere_On_Chain( state.get() ) )
    {
        float force = Simulation_Get_MaxTorqueNorm( state.get() );

        iteration_history[force_index] = Simulation_Get_Iteration( state.get() );

        force_history[force_index] = force;
        ++force_index;
        force_index = force_index % history_size;

        if( force < force_min )
            force_min = force;
        if( force > force_max )
            force_max = force;
    }

    if( !show_ )
        return;

    ImGui::SetNextWindowSizeConstraints( { 500, 300 }, { 800, 999999 } );

    ImGui::Begin( "Plots", &show_ );
    {
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
        if( ImGui::BeginTabBar( "plots_tab_bar", tab_bar_flags ) )
        {
            if( ImGui::BeginTabItem( "Energy" ) )
            {
                int noi = Chain_Get_NOI( state.get() );
                if( noi != energies.size() )
                {
                    rx.resize( noi );
                    energies.resize( noi );
                }

                int size_interp = noi + ( noi - 1 ) * n_interpolate;
                if( size_interp != energies_interpolated.size() )
                {
                    rx_interpolated.resize( size_interp );
                    energies_interpolated.resize( size_interp );
                }

                Chain_Get_Rx( state.get(), rx.data() );
                Chain_Get_Energy( state.get(), energies.data() );

                if( noi > 1 )
                {
                    Chain_Get_Rx_Interpolated( state.get(), rx_interpolated.data() );
                    Chain_Get_Energy_Interpolated( state.get(), energies_interpolated.data() );
                }

                std::string overlay = fmt::format( "{:.3e}", energies[System_Get_Index( state.get() )] );

                if( ImPlot::BeginPlot(
                        "", "Rx", "E [meV]",
                        ImVec2(
                            ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                            ImGui::GetWindowContentRegionMax().y - 110.f ) ) )
                {
                    if( plot_image_energies )
                    {
                        ImPlot::PlotScatter( "", rx.data(), energies.data(), noi );
                    }
                    if( plot_interpolated_energies && noi > 1 )
                    {
                        ImPlot::PlotLine( "", rx_interpolated.data(), energies_interpolated.data(), size_interp );
                    }

                    ImPlot::EndPlot();
                }

                ImGui::Checkbox( "Image energies", &plot_image_energies );
                ImGui::Checkbox( "Interpolated energies", &plot_interpolated_energies );
                ImGui::SameLine();
                ImGui::PushItemWidth( 100 );
                if( ImGui::InputInt( "##energies_n_interp", &n_interpolate ) )
                {
                    if( n_interpolate > 1 )
                        Parameters_GNEB_Set_N_Energy_Interpolations( state.get(), n_interpolate );
                    else
                        n_interpolate = 1;
                }
                ImGui::PopItemWidth();

                ImGui::EndTabItem();
            }

            if( ImGui::BeginTabItem( "Convergence" ) )
            {
                ImPlot::FitNextPlotAxes();
                if( ImPlot::BeginPlot(
                        "", "iteration", "max(F)",
                        ImVec2(
                            ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                            ImGui::GetWindowContentRegionMax().y - 90.f ) ) )
                {
                    ImPlot::PlotLine( "", iteration_history.data(), force_history.data(), history_size, force_index );
                    ImPlot::EndPlot();
                }

                if( ImGui::InputInt( "history size", &history_size, 10, 100 ) )
                {
                    if( history_size < 3 )
                        history_size = 3;
                    iteration_history.resize( history_size );
                    force_history.resize( history_size );
                }

                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void PlotsWidget::update_data() {}

} // namespace ui