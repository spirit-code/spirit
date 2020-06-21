#pragma once
#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <memory>

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>

struct State;

class PlotWidget : public QtCharts::QChartView
{

public:
    PlotWidget(std::shared_ptr<State> state, bool plot_image_energies=true, bool plot_interpolated=false);
    void updateData();

    bool plot_image_energies;
    bool plot_interpolated;
    int  plot_interpolated_n;

private:
    std::shared_ptr<State> state;
    std::vector<float> Rx, energies;
    std::vector<float> Rx_interp, energies_interp;

    QtCharts::QChart * chart;
    QtCharts::QScatterSeries * series_E_normal;
    QtCharts::QScatterSeries * series_E_climbing;
    QtCharts::QScatterSeries * series_E_falling;
    QtCharts::QScatterSeries * series_E_stationary;
    QtCharts::QScatterSeries * series_E_current;
    QtCharts::QLineSeries * series_E_interp;

    QImage triangleUpRed, triangleUpBlue, triangleDownRed, triangleDownBlue;

    void plotEnergies();
};

#endif