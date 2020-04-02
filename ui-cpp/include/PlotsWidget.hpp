#pragma once
#ifndef SPIRIT_PLOTSWIDGET_HPP
#define SPIRIT_PLOTSWIDGET_HPP

#include "ui_PlotsWidget.h"

#include "PlotWidget.hpp"

#include <QWidget>

#include <memory>

struct State;

class PlotsWidget : public QWidget, private Ui::PlotsWidget
{
    Q_OBJECT

public:
    PlotsWidget( std::shared_ptr<State> state );

    std::shared_ptr<State> state;
    PlotWidget * energyPlot;

private slots:
    void updatePlotData();
    void refreshClicked();
    void updatePlotSettings();
};

#endif