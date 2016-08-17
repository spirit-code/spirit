#include "PlotWidget.h"
#include "Manifoldmath.h"
#include "gr.h"

#include "Interface_Chain.h"
#include "Interface_System.h"

#include <vector>

PlotWidget::PlotWidget(std::shared_ptr<State> state)
{
	this->state = state;
}


void PlotWidget::draw()
{
    // TODO: how to get automatic axes min/max??
    gr_setwindow(0, 10, -25, 10);
    gr_setviewport(0.1, 0.9, 0.1, 0.9);

    gr_setlinewidth(1);
    gr_setlinecolorind(1);
    gr_grid(0.5, 0.5, 0, 0, 4, 4);
    gr_axes(0.5, 0.5, 0, 0, 4, 4, 0.01);

    if( Chain_Get_NOI(state.get()) > 1 ) this->plotEnergiesInterpolated();
    this->plotEnergies();
}

void PlotWidget::plotEnergies()
{
    std::vector<double> x(0);
    std::vector<double> y(0);

    for (int i = 0; i < Chain_Get_NOI(state.get()); ++i)
	{
        x.push_back(i);
        y.push_back(System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i));
        // std::cerr << System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i) << std::endl;
    }
    
    gr_setmarkercolorind(75);
    gr_setmarkertype(-1);
    gr_setmarkersize(1.5);
    gr_polymarker(x.size(), x.data(), y.data());
}

void PlotWidget::plotEnergiesInterpolated()
{
    std::vector<double> x(0);
    std::vector<double> y(0);

    // TODO: get true interpolated values
    for (int i = 0; i < Chain_Get_NOI(state.get()); ++i)
	{
        x.push_back(i);
        y.push_back(System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i));
        // std::cerr << System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i) << std::endl;
    }
    
    gr_setlinewidth(4);
    gr_setlinecolorind(75);
    gr_spline(x.size(), x.data(), y.data(), 500, 0);
}