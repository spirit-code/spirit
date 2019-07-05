#include <AdvancedGraph.hpp>

#include <Spirit/System.h>
#include <Spirit/Chain.h>
#include <Spirit/Parameters_GNEB.h>

#include <memory>

class EnergyGraph : public AdvancedGraph
{
public:
    EnergyGraph(Widget * parent, std::shared_ptr<State> state);

    void updateData()
    {
        int noi = Chain_Get_NOI(state.get());
        int nos = System_Get_NOS(state.get());

        if( this->plot_interpolated && this->plot_interpolated_n != Parameters_GNEB_Get_N_Energy_Interpolations(state.get()) )
            Parameters_GNEB_Set_N_Energy_Interpolations(state.get(), this->plot_interpolated_n);

        int size_interp = noi + (noi - 1)*Parameters_GNEB_Get_N_Energy_Interpolations(state.get());

        // Allocate arrays
        auto Rx = std::vector<float>(noi, 0);
        auto energies = std::vector<float>(noi, 0);
        auto Rx_interp = std::vector<float>(size_interp, 0);
        auto energies_interp = std::vector<float>(size_interp, 0);

        // Get Data
        float Rx_tot = System_Get_Rx(state.get(), noi - 1);
        Chain_Get_Rx(state.get(), Rx.data());
        Chain_Get_Energy(state.get(), energies.data());
        if( this->plot_interpolated )
        {
            Chain_Get_Rx_Interpolated(state.get(), Rx_interp.data());
            Chain_Get_Energy_Interpolated(state.get(), energies_interp.data());
        }

        // TODO: set data values
        float E_min=1e8, E_max=-1e8;
        int idx_current = System_Get_Index(state.get());
        if( this->plot_image_energies )
        {
            for( int i = 0; i < noi; ++i )
            {
                if( i > 0 && Rx_tot > 0 )
                    Rx[i] = Rx[i] / Rx_tot;
                energies[i] = energies[i] / nos;
                if( energies[i] > E_max )
                    E_max = energies[i];
                if( energies[i] < E_min )
                    E_min = energies[i];

                if( i != idx_current )
                {
                    this->marker_colors[i] = nanogui::Color(55,126,184,255);
                    this->marker_scale[i] = this->default_marker_scale;
                } else {
                    this->marker_colors[i] = nanogui::Color(228,26,28,255);
                    this->marker_scale[i] = this->default_marker_scale * 1.2;
                }
            }
        }

        this->x_values  = Rx;
        this->y_values  = energies;

        float delta = 0.1*(E_max - E_min);
        if (delta < 1e-6)
            delta = 0.1;
        this->my_min = E_min - delta;
        this->my_max = E_max + delta;

        if( this->x_values.size() != this->markers.size() )
        {
            this->markers.resize(x_values.size(), this->default_marker);
            this->marker_scale.resize(x_values.size(), this->default_marker_scale);
            this->marker_colors.resize(x_values.size(), this->default_marker_color);
        }
    }

private:
    std::shared_ptr<State> state;

    bool plot_image_energies;
    bool plot_interpolated;
    int plot_interpolated_n;
};