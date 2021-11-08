#pragma once
#ifndef SPIRIT_SETTINGSWIDGET_HPP
#define SPIRIT_SETTINGSWIDGET_HPP

#include "ui_SettingsWidget.h"

#include "ConfigurationsWidget.hpp"
#include "GeometryWidget.hpp"
#include "HamiltonianGaussianWidget.hpp"
#include "HamiltonianHeisenbergWidget.hpp"
#include "ParametersWidget.hpp"
#include "VisualisationSettingsWidget.hpp"

#include <QtWidgets/QWidget>

#include <memory>

class SpinWidget;
struct State;

class SettingsWidget : public QWidget, private Ui::SettingsWidget
{
    Q_OBJECT

public:
    SettingsWidget( std::shared_ptr<State> state, SpinWidget * spinWidget );
    void SelectTab( int index );
    void incrementNCellStep( int increment );
    void toggleGeometry();

    std::shared_ptr<State> state;

public slots:
    void updateData();
    // Configurations
    void configurationAddNoise();
    void randomPressed();
    void lastConfiguration();

private:
    SpinWidget * _spinWidget;
    ConfigurationsWidget * configurationsWidget;
    ParametersWidget * parametersWidget;
    HamiltonianHeisenbergWidget * hamiltonianHeisenbergWidget;
    HamiltonianGaussianWidget * hamiltonianGaussianWidget;
    GeometryWidget * geometryWidget;
    VisualisationSettingsWidget * visualisationSettingsWidget;
};

#endif