#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QWidget>

#include <memory>

#include "ConfigurationsWidget.hpp"
#include "ParametersWidget.hpp"
#include "HamiltonianHeisenbergNeighboursWidget.hpp"
#include "HamiltonianHeisenbergPairsWidget.hpp"
#include "HamiltonianGaussianWidget.hpp"
#include "VisualisationSettingsWidget.hpp"

#include "ui_SettingsWidget.h"

class SpinWidget;
struct State;

class SettingsWidget : public QWidget, private Ui::SettingsWidget
{
    Q_OBJECT

public:
	SettingsWidget(std::shared_ptr<State> state, SpinWidget *spinWidget);
	void updateData();
	void SelectTab(int index);
	void incrementNCellStep(int increment);

	std::shared_ptr<State> state;

public slots:
	// Configurations
	void configurationAddNoise();
	void randomPressed();
	void lastConfiguration();

private:
	SpinWidget *_spinWidget;
	ConfigurationsWidget * configurationsWidget;
	ParametersWidget * parametersWidget;
	HamiltonianHeisenbergNeighboursWidget * hamiltonianHeisenbergNeighboursWidget;
	HamiltonianHeisenbergPairsWidget * hamiltonianHeisenbergPairsWidget;
	HamiltonianGaussianWidget * hamiltonianGaussianWidget;
	VisualisationSettingsWidget * visualisationSettingsWidget;
};

#endif