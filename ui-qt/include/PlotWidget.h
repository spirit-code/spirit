#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H


#include <memory>

#include <QWidget>

#include "Spin_System_Chain.h"
#include "Interface_State.h"


class PlotWidget : public QWidget	// We need a proper 2D plotting solution!!
{

public:
	PlotWidget(std::shared_ptr<State> state);
	void update();

private:
	std::shared_ptr<State> state;

};

#endif