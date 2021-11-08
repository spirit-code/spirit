#include <plots_widget.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <imgui/imgui.h>

#include <implot/implot.h>

#include <fmt/format.h>

#include <cmath>
#include <string>
#include <vector>

namespace ui
{

void plot_tooltip( const char * label_id, const float * xs, const float * ys, int count )
{
    if( ImPlot::IsPlotHovered() )
    {
        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        auto mouse_pixels = ImPlot::PlotToPixels( mouse.x, mouse.y, IMPLOT_AUTO );

        int idx_best        = 0;
        float distance_best = 1e30f;
        for( int idx = 0; idx < count; ++idx )
        {
            auto data_pixels = ImPlot::PlotToPixels( xs[idx], ys[idx], IMPLOT_AUTO );

            float dx       = mouse_pixels.x - data_pixels.x;
            float dy       = mouse_pixels.y - data_pixels.y;
            float distance = std::sqrt( dx * dx + dy * dy );
            if( distance < distance_best )
            {
                idx_best      = idx;
                distance_best = distance;
            }
        }

        ImGui::SetNextWindowPos( ImPlot::PlotToPixels( xs[idx_best], ys[idx_best], IMPLOT_AUTO ) );
        ImGui::BeginTooltip();
        ImGui::Text( "(%.2f, %.2f)", xs[idx_best], ys[idx_best] );
        ImGui::EndTooltip();
    }
}

PlotsWidget::PlotsWidget( bool & show, std::shared_ptr<State> state ) : WidgetBase( show ), state( state )
{
    title             = "Plots";
    history_size      = 200;
    iteration_history = std::vector<float>( history_size );
    force_history     = std::vector<float>( history_size );
}

void PlotsWidget::hook_pre_show()
{
    if( Simulation_Running_Anywhere_On_Chain( state.get() ) )
    {
        iteration_history[force_index] = Simulation_Get_Iteration( state.get() );
        force_history[force_index]     = Simulation_Get_MaxTorqueNorm( state.get() );
        ++force_index;
        force_index = force_index % history_size;
    }
}

void PlotsWidget::show_content()
{
    auto & style = ImGui::GetStyle();

    static bool plot_image_energies        = true;
    static bool plot_interpolated_energies = true;

    static bool fit_axes = false;
    static bool tooltip  = false;

    static int n_interpolate = Parameters_GNEB_Get_N_Energy_Interpolations( state.get() );
    static std::vector<float> rx( 1, 0 );
    static std::vector<float> energies( 1, 0 );
    static std::vector<float> rx_interpolated( 1, 0 );
    static std::vector<float> energies_interpolated( 1, 0 );
    static std::vector<float> max_force( 1, 0 );

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

            Chain_Get_Rx( state.get(), rx.data() );
            Chain_Get_Energy( state.get(), energies.data() );

            int idx_current = System_Get_Index( state.get() );
            ImGui::TextUnformatted(
                fmt::format( "E[{}] = {:.5e} meV", idx_current + 1, energies[idx_current] ).c_str() );

            if( fit_axes )
                ImPlot::FitNextPlotAxes();
            if( ImPlot::BeginPlot(
                    "", "Rx", "E [meV]",
                    ImVec2(
                        ImGui::GetWindowContentRegionMax().x - 2 * style.FramePadding.x,
                        ImGui::GetWindowContentRegionMax().y - 130.f ),
                    ImPlotFlags_NoMousePos ) )
            {
                // Line plots
                if( noi > 1 )
                {
                    if( plot_interpolated_energies )
                    {
                        int size_interp = noi + ( noi - 1 ) * n_interpolate;
                        if( size_interp != energies_interpolated.size() )
                        {
                            rx_interpolated.resize( size_interp );
                            energies_interpolated.resize( size_interp );
                        }

                        Chain_Get_Rx_Interpolated( state.get(), rx_interpolated.data() );
                        Chain_Get_Energy_Interpolated( state.get(), energies_interpolated.data() );

                        ImPlot::SetNextLineStyle( ImVec4( ImColor( 65, 105, 225 ) ) );
                        ImPlot::PlotLine( "", rx_interpolated.data(), energies_interpolated.data(), size_interp );
                    }
                }

                // Scatter plots
                if( plot_image_energies )
                {
                    // Image energies except current image
                    if( noi > 1 )
                    {
                        std::vector<float> rx_regular( 0 );
                        std::vector<float> e_regular( 0 );
                        std::vector<float> rx_climbing( 0 );
                        std::vector<float> e_climbing( 0 );
                        std::vector<float> rx_falling( 0 );
                        std::vector<float> e_falling( 0 );
                        std::vector<float> rx_stationary( 0 );
                        std::vector<float> e_stationary( 0 );

                        if( noi != max_force.size() )
                            max_force.resize( noi );

                        Simulation_Get_Chain_MaxTorqueNorms( state.get(), max_force.data() );
                        int idx_max_force = idx_current;
                        float max_f       = max_force[idx_current];

                        // Get max. force image
                        for( int idx = 0; idx < noi; ++idx )
                        {
                            if( max_force[idx] > max_f )
                            {
                                idx_max_force = idx;
                                max_f         = max_force[idx];
                            }
                        }
                        // Distinguish image types
                        for( int idx = 0; idx < noi; ++idx )
                        {
                            if( idx != idx_current && idx != idx_max_force )
                            {
                                if( Parameters_GNEB_Get_Climbing_Falling( state.get(), idx ) == GNEB_IMAGE_NORMAL )
                                {
                                    rx_regular.push_back( rx[idx] );
                                    e_regular.push_back( energies[idx] );
                                }
                                else if(
                                    Parameters_GNEB_Get_Climbing_Falling( state.get(), idx ) == GNEB_IMAGE_CLIMBING )
                                {
                                    rx_climbing.push_back( rx[idx] );
                                    e_climbing.push_back( energies[idx] );
                                }
                                else if(
                                    Parameters_GNEB_Get_Climbing_Falling( state.get(), idx ) == GNEB_IMAGE_FALLING )
                                {
                                    rx_falling.push_back( rx[idx] );
                                    e_falling.push_back( energies[idx] );
                                }
                                else if(
                                    Parameters_GNEB_Get_Climbing_Falling( state.get(), idx ) == GNEB_IMAGE_STATIONARY )
                                {
                                    rx_stationary.push_back( rx[idx] );
                                    e_stationary.push_back( energies[idx] );
                                }
                            }
                        }

                        int idx_max_energy = 0;
                        float max_energy   = energies[0];
                        int max_image_type = Parameters_GNEB_Get_Climbing_Falling( state.get(), 0 );
                        for( int idx = 1; idx < noi; ++idx )
                        {
                            if( energies[idx] > max_energy )
                            {
                                idx_max_energy = idx;
                                max_image_type = Parameters_GNEB_Get_Climbing_Falling( state.get(), idx );
                            }
                        }

                        // Regular image dots (blue)
                        ImPlot::SetNextMarkerStyle(
                            IMPLOT_AUTO, IMPLOT_AUTO, ImVec4( ImColor( 65, 105, 225 ) ), IMPLOT_AUTO,
                            ImVec4( ImColor( 65, 105, 225 ) ) );
                        ImPlot::PlotScatter( "", rx_regular.data(), e_regular.data(), rx_regular.size() );

                        // Climbing image triangles
                        ImPlot::SetNextMarkerStyle(
                            ImPlotMarker_Up, IMPLOT_AUTO, ImVec4( ImColor( 65, 105, 225 ) ), IMPLOT_AUTO,
                            ImVec4( ImColor( 65, 105, 225 ) ) );
                        ImPlot::PlotScatter( "", rx_climbing.data(), e_climbing.data(), rx_climbing.size() );

                        // Falling image triangles
                        ImPlot::SetNextMarkerStyle(
                            ImPlotMarker_Down, IMPLOT_AUTO, ImVec4( ImColor( 65, 105, 225 ) ), IMPLOT_AUTO,
                            ImVec4( ImColor( 65, 105, 225 ) ) );
                        ImPlot::PlotScatter( "", rx_falling.data(), e_falling.data(), rx_falling.size() );

                        // Stationary image squares
                        ImPlot::SetNextMarkerStyle(
                            ImPlotMarker_Square, IMPLOT_AUTO, ImVec4( ImColor( 65, 105, 225 ) ), IMPLOT_AUTO,
                            ImVec4( ImColor( 65, 105, 225 ) ) );
                        ImPlot::PlotScatter( "", rx_stationary.data(), e_stationary.data(), rx_stationary.size() );

                        // Max force image marker (orange)
                        if( idx_max_force != idx_current )
                        {
                            int max_force_image_type
                                = Parameters_GNEB_Get_Climbing_Falling( state.get(), idx_max_force );
                            ImPlotMarker marker_style = IMPLOT_AUTO;
                            if( max_force_image_type == GNEB_IMAGE_CLIMBING )
                                marker_style = ImPlotMarker_Up;
                            if( max_force_image_type == GNEB_IMAGE_FALLING )
                                marker_style = ImPlotMarker_Down;
                            if( max_force_image_type == GNEB_IMAGE_STATIONARY )
                                marker_style = ImPlotMarker_Square;

                            ImPlot::SetNextMarkerStyle(
                                marker_style, IMPLOT_AUTO, ImVec4( ImColor( 255, 165, 0 ) ), IMPLOT_AUTO,
                                ImVec4( ImColor( 255, 165, 0 ) ) );
                            ImPlot::PlotScatter( "", &rx[idx_max_force], &energies[idx_max_force], 1 );
                        }
                    }

                    // Current image marker (red)
                    float rx_current          = rx[idx_current];
                    float energy_current      = energies[idx_current];
                    int current_image_type    = Parameters_GNEB_Get_Climbing_Falling( state.get(), idx_current );
                    ImPlotMarker marker_style = IMPLOT_AUTO;
                    if( current_image_type == GNEB_IMAGE_CLIMBING )
                        marker_style = ImPlotMarker_Up;
                    if( current_image_type == GNEB_IMAGE_FALLING )
                        marker_style = ImPlotMarker_Down;
                    if( current_image_type == GNEB_IMAGE_STATIONARY )
                        marker_style = ImPlotMarker_Square;
                    ImPlot::SetNextMarkerStyle(
                        marker_style, IMPLOT_AUTO, ImVec4( ImColor( 255, 0, 0 ) ), IMPLOT_AUTO,
                        ImVec4( ImColor( 255, 0, 0 ) ) );
                    ImPlot::PlotScatter( "", &rx_current, &energy_current, 1 );
                }

                if( tooltip )
                    plot_tooltip( "tooltip", rx.data(), energies.data(), rx.size() );

                ImPlot::EndPlot();
            }

            ImGui::Checkbox( "Image energies", &plot_image_energies );
            ImGui::SameLine();
            ImGui::Checkbox( "Autofit axes", &fit_axes );
            ImGui::SameLine();
            ImGui::Checkbox( "Tooltip", &tooltip );

            ImGui::Checkbox( "Interpolated energies", &plot_interpolated_energies );
            ImGui::SameLine();
            ImGui::PushItemWidth( 100 );
            if( ImGui::InputInt( "##energies_n_interp", &n_interpolate, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue ) )
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
            ImGui::TextUnformatted( fmt::format( "Latest: {:.5e}", force_history[force_index] ).c_str() );
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

            if( ImGui::InputInt( "history size", &history_size, 10, 100, ImGuiInputTextFlags_EnterReturnsTrue ) )
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

void PlotsWidget::update_data() {}

} // namespace ui