#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <memory>

#include <QtCharts/QChartView>

struct State;

class PlotWidget : public QtCharts::QChartView	// We need a proper 2D plotting solution!!
{

public:
	PlotWidget(std::shared_ptr<State> state);
	void update();

private:
	std::shared_ptr<State> state;

};

#endif