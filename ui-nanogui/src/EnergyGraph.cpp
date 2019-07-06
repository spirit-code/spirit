#include <EnergyGraph.hpp>

using Color = nanogui::Color;

EnergyGraph::EnergyGraph(Widget * parent, std::shared_ptr<State> state)
    : AdvancedGraph(parent, Marker::CIRCLE, nanogui::Color(0,0,255,255), 1.4), state(state),
    plot_image_energies(true), plot_interpolated(false), plot_interpolated_n(10)
{
    this->setGrid(true);
    this->setMarginBot(40);
    this->setXLabel("Reaction Coordinate");
    this->setYLabel("Energy [meV]");
    this->setXMin(0);
    this->setXMax(1);
    this->setNTicksX(5);
    this->setNTicksY(5);
    this->setYMin(-0.1);
    this->setYMax(0.1);
    this->setLine(true);
    this->setValues( {0}, {0} );

    this->mx_min    = 0;
    this->mx_max    = 1;
};