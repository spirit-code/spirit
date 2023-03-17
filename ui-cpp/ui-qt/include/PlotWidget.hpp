#pragma once
#ifndef SPIRIT_PLOTWIDGET_HPP
#define SPIRIT_PLOTWIDGET_HPP

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>

#include <Spirit/Spirit_Defines.h>

#include <memory>

struct State;

class PlotWidget : public QtCharts::QChartView
{

public:
    PlotWidget( std::shared_ptr<State> state, bool plot_image_energies = true, bool plot_interpolated = false );
    void updateData();

    bool plot_image_energies;
    bool plot_interpolated;
    int plot_interpolated_n;
    bool divide_by_nos;
    bool renormalize_Rx;

private:
    std::shared_ptr<State> state;
    std::vector<scalar> Rx, energies;
    std::vector<scalar> Rx_interp, energies_interp;

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