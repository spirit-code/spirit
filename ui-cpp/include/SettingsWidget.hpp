#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QtWidgets/QWidget>

#include <memory>

#include "ConfigurationsWidget.hpp"
#include "ParametersWidget.hpp"
#include "HamiltonianHeisenbergWidget.hpp"
#include "HamiltonianMicromagneticWidget.hpp"
#include "HamiltonianGaussianWidget.hpp"
#include "GeometryWidget.hpp"
#include "VisualisationSettingsWidget.hpp"

#include "ui_SettingsWidget.h"

class SpinWidget;
struct State;

class SettingsWidget : public QWidget, private Ui::SettingsWidget
{
    Q_OBJECT

public:
    SettingsWidget(std::shared_ptr<State> state, SpinWidget *spinWidget);
    void SelectTab(int index);
    void incrementNCellStep(int increment);
    void toggleGeometry();

    std::shared_ptr<State> state;

public slots:
    void updateData();
    // Configurations
    void configurationAddNoise();
    void randomPressed();
    void lastConfiguration();

    void updateHamiltonian(Hamiltonian_Type);

private:
    SpinWidget *_spinWidget;
    ConfigurationsWidget * configurationsWidget;
    ParametersWidget * parametersWidget;
    HamiltonianHeisenbergWidget * hamiltonianHeisenbergWidget;
    HamiltonianMicromagneticWidget * hamiltonianMicromagneticWidget;
    HamiltonianGaussianWidget * hamiltonianGaussianWidget;
    GeometryWidget * geometryWidget;
    VisualisationSettingsWidget * visualisationSettingsWidget;
};

#endif