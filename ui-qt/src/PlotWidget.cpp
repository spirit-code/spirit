#include "PlotWidget.h"

#include "Interface_Chain.h"
#include "Interface_System.h"


using namespace QtCharts;

PlotWidget::PlotWidget(std::shared_ptr<State> state, bool plot_interpolated) :
	plot_interpolated(plot_interpolated)
{
	this->state = state;

    // Create Chart
	chart = new QChart();
	chart->legend()->hide();
	chart->setTitle("Simple line chart example");

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


void PlotWidget::update()
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
		*series_E << QPointF(System_Get_Rx(state.get(), i)/System_Get_Rx(state.get(), noi-1), System_Get_Energy(state.get(), i) / System_Get_NOS(state.get(), i));
		// std::cerr << System_Get_Energy(state.get(), i)/System_Get_NOS(state.get(), i) << std::endl;
	}
	// Re-add Series to chart
	chart->removeSeries(series_E);
	chart->addSeries(series_E);

	// Current image - red dot
	series_E_current->clear();
	int i = System_Get_Index(state.get());
	*series_E_current << QPointF(System_Get_Rx(state.get(), i)/System_Get_Rx(state.get(), noi-1), System_Get_Energy(state.get(), i) / System_Get_NOS(state.get(), i));
	chart->removeSeries(series_E_current);
	chart->addSeries(series_E_current);

	// Renew axes
	this->chart->createDefaultAxes();
}

void PlotWidget::plotEnergiesInterpolated()
{
	// TODO: this seems incredibly inefficient, how can we do better??
	// Clear series
	series_E_interp->clear();

	// Add data to series
	int noi = Chain_Get_NOI(state.get());
	int nos = System_Get_NOS(state.get());
	auto Rx = Chain_Get_Rx_Interpolated(state.get());
	auto E = Chain_Get_Energy_Interpolated(state.get());
	for (unsigned int i = 0; i < Rx.size(); ++i)
	{
		*series_E_interp << QPointF(Rx[i]/Rx.back(), E[i] / nos);
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