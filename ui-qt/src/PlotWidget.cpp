#include "PlotWidget.h"
#include "Manifoldmath.h"
#include "gr.h"

PlotWidget::PlotWidget(std::shared_ptr<State> state)
{
	this->state = state;

}


void PlotWidget::draw()
{
    double x[] = {1, 3, 5, 7, 9};
    double y[] = {1, 9, 2, 5, 4};

    gr_setwindow(0, 10, 0, 10);
    gr_setviewport(0.1, 0.9, 0.1, 0.9);

    gr_setlinewidth(1);
    gr_setlinecolorind(1);
    gr_grid(0.5, 0.5, 0, 0, 4, 4);
    gr_axes(0.5, 0.5, 0, 0, 4, 4, 0.01);
    gr_setlinewidth(5);
    gr_setlinecolorind(75);
    gr_spline(sizeof(x) / sizeof(double), x, y, 500, 0);
}

