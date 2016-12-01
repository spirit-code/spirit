#include "PlotWidget.hpp"

#include "Interface_Chain.h"
#include "Interface_System.h"
#include "Interface_Parameters.h"


using namespace QtCharts;

PlotWidget::PlotWidget(std::shared_ptr<State> state, bool plot_interpolated) :
	plot_interpolated(plot_interpolated)
{
	this->state = state;

    // Create Chart
	chart = new QChart();
	chart->legend()->hide();
	chart->setTitle("");
	// chart->axisX()->setTitleText("Rx");
	// chart->axisY()->setTitleText("E");

    // Use Chart
	this->setChart(chart);
	this->setRenderHint(QPainter::Antialiasing);

	// Create Series
	series_E = new QScatterSeries();
	series_E_current = new QScatterSeries();
	series_E_current->setColor(QColor("red"));
	series_E_interp = new QLineSeries();
	// Add Series
	chart->addSeries(series_E);
	chart->addSeries(series_E_current);
	//chart->addSeries(series_E_interp);

	// Axes
	this->chart->createDefaultAxes();
}


void PlotWidget::updateData()
{
    if( Chain_Get_NOI(state.get()) > 1 && this->plot_interpolated ) this->plotEnergiesInterpolated();
    this->plotEnergies();
	this->chart->update();
}

void PlotWidget::plotEnergies()
{
	// TODO: this seems incredibly inefficient, how can we do better??
	// Clear series
	series_E->clear();

	// Add data to series
	int noi = Chain_Get_NOI(state.get());
	for (int i = 0; i < noi; ++i)
	{
		float x = 0;
		float Rx_tot = System_Get_Rx(state.get(), noi - 1);
		if (i > 0 && Rx_tot > 0) x = System_Get_Rx(state.get(), i) / Rx_tot;
		*series_E << QPointF(x, System_Get_Energy(state.get(), i) / System_Get_NOS(state.get(), i));
		// std::cerr << System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i) << std::endl;
	}
	// Re-add Series to chart
	chart->removeSeries(series_E);
	chart->addSeries(series_E);

	// Current image - red dot
	series_E_current->clear();
	int i = System_Get_Index(state.get());
	float x = 0;
	float Rx_tot = System_Get_Rx(state.get(), noi - 1);
	if (i > 0 && Rx_tot > 0) x = System_Get_Rx(state.get(), i) / Rx_tot;
	*series_E_current << QPointF(x, System_Get_Energy(state.get(), i) / System_Get_NOS(state.get(), i));
	chart->removeSeries(series_E_current);
	chart->addSeries(series_E_current);

	// Renew axes
	this->chart->createDefaultAxes();
}

void PlotWidget::plotEnergiesInterpolated()
{
	if (Chain_Get_NOI(state.get()) <= 1) return;
	// TODO: this seems incredibly inefficient, how can we do better??
	// Clear series
	series_E_interp->clear();

	// Add data to series
	int noi = Chain_Get_NOI(state.get());
	int nos = System_Get_NOS(state.get());
	int size_interp = (noi-1)*Parameters_Get_GNEB_N_Energy_Interpolations(state.get());
	float *Rx = new float[size_interp];
	float *E = new float[size_interp];
	Chain_Get_Rx_Interpolated(state.get(), Rx);
	Chain_Get_Energy_Interpolated(state.get(), E);
	for (int i = 0; i < size_interp; ++i)
	{
		*series_E_interp << QPointF(Rx[i]/Rx[size_interp-1], E[i] / nos);
	}

	// Re-add Series to chart
	chart->removeSeries(series_E_interp);
	chart->addSeries(series_E_interp);
	
	// Renew axes
	this->chart->createDefaultAxes();
}

void PlotWidget::set_interpolating(bool b)
{
	if (this->plot_interpolated == b) return;
	else if (this->plot_interpolated && !b)
	{
		// Turn interpolation off
		chart->removeSeries(series_E_interp);
	}
	else if (!this->plot_interpolated && b)
	{
		// Turn interpolation on
		chart->addSeries(series_E_interp);
	}
	this->plot_interpolated = b;
}