#pragma once
#ifndef PLOTSWIDGET_H
#define PLOTSWIDGET_H

#include <QtWidgets>

#include <memory>

#include "Spin_System_Chain.h"
#include "PlotWidget.h"
#include "Interface_State.h"

#include "ui_PlotsWidget.h"

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