#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <memory>

#include <QWidget>

#include "Spin_System_Chain.h"

#include "grwidget.h"

struct State;


class PlotWidget : public GRWidget	// We need a proper 2D plotting solution!!
{

public:
	PlotWidget(std::shared_ptr<State> state);
	void draw();

private:
	std::shared_ptr<State> state;
	void plotEnergies();
	void plotEnergiesInterpolated();
};

#endif
