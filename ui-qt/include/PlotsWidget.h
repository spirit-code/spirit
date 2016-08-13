#pragma once
#ifndef PLOTSWIDGET_H
#define PLOTSWIDGET_H

#include <QWidget>

#include <memory>

#include "PlotWidget.h"

#include "ui_PlotsWidget.h"

struct State;

class PlotsWidget : public QWidget, private Ui::PlotsWidget
{
    Q_OBJECT

public:
	PlotsWidget(std::shared_ptr<State> state);

	std::shared_ptr<State> state;
    
    PlotWidget * energyPlot;

private slots:
	void RefreshClicked();
	void ChangeInterpolationClicked();


};

#endif