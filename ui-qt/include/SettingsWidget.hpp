#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QWidget>

#include <memory>

#include "ConfigurationsWidget.hpp"
#include "ParametersWidget.hpp"
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
	// Setup Input Validators
	void Setup_Input_Validators();
	// Setup Slots
	void Setup_Hamiltonian_Isotropic_Slots();
	void Setup_Hamiltonian_Anisotropic_Slots();
	// Load a set of parameters from the spin systems
	void Load_Hamiltonian_Isotropic_Contents();
	void Load_Hamiltonian_Anisotropic_Contents();
	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;
	SpinWidget *_spinWidget;
	ConfigurationsWidget * configurationsWidget;
	ParametersWidget * parametersWidget;
	VisualisationSettingsWidget * visualisationSettingsWidget;

private slots:
	// Configurations
	void set_hamiltonian_iso();
	// void set_hamiltonian_aniso();
	void set_hamiltonian_aniso_bc();
	void set_hamiltonian_aniso_mu_s();
	void set_hamiltonian_aniso_field();
	void set_hamiltonian_aniso_ani();
};

#endif