#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <memory>

#include <QWidget>

struct State;

class PlotWidget : public QWidget	// We need a proper 2D plotting solution!!
{

public:
	PlotWidget(std::shared_ptr<State> state);
	void update();

private:
	std::shared_ptr<State> state;

};

#endif