#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H


#include <memory>

#include <QWidget>

#include "Spin_System_Chain.h"

#include "grwidget.h"


class PlotWidget : public GRWidget	// We need a proper 2D plotting solution!!
{

public:
	PlotWidget(std::shared_ptr<Data::Spin_System_Chain> c);
	void draw();

private:
	std::shared_ptr<Data::Spin_System_Chain> c;

};

#endif
