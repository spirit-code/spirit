#include <EnergyGraph.hpp>

using Color = nanogui::Color;

EnergyGraph::EnergyGraph(Widget * parent, std::shared_ptr<State> state)
    : AdvancedGraph(parent, Marker::SQUARE, nanogui::Color(0,0,255,255), 1.4), state(state),
    plot_image_energies(true), plot_interpolated(false), plot_interpolated_n(10)
{
    this->setSize({300, 200});
    this->setPosition({0,0});
    this->setGrid(true);
    this->setMarginBot(40);
    this->setXLabel("Reaction Coordinate");
    this->setYLabel("Energy [meV]");
    this->setXMin(0);
    this->setXMax(9.5);
    this->setNTicksX(10);
    this->setYMin(1e6); 
    this->setYMax(5e6);
    this->setLine(true);
    this->setValues( {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1.2e6, 2.2e6, 3e6, 4.02e6, 2e6, 4.5e6, 3.7e6, 3.8e6, 2.9e6, 10e6} );
};