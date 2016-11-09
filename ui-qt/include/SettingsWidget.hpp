#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QWidget>

#include <memory>

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

	std::shared_ptr<State> state;

private:
	// Setup Input Validators
	void Setup_Input_Validators();
	// Setup Slots
	void Setup_Configurations_Slots();
	void Setup_Transitions_Slots();
	void Setup_Hamiltonian_Isotropic_Slots();
	void Setup_Hamiltonian_Anisotropic_Slots();
	void Setup_Parameters_Slots();
	void Setup_Visualization_Slots();
	// Load a set of parameters from the spin systems
	void Load_Hamiltonian_Isotropic_Contents();
	void Load_Hamiltonian_Anisotropic_Contents();
	void Load_Parameters_Contents();
	void Load_Visualization_Contents();
	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	SpinWidget *_spinWidget;

private slots:
	// Parameters
	void set_parameters();
	// Configurations
	void set_hamiltonian_iso();
	// void set_hamiltonian_aniso();
	void set_hamiltonian_aniso_bc();
	void set_hamiltonian_aniso_mu_s();
	void set_hamiltonian_aniso_field();
	void set_hamiltonian_aniso_ani();
	void set_hamiltonian_aniso_stt();
	void set_hamiltonian_aniso_temp();
	// Visualization
	void set_visualization();
	// Configurations
	void configurationAddNoise();
	void randomPressed();
	void addNoisePressed();
	void domainWallPressed();
	void plusZ();
	void minusZ();
	void create_Hopfion();
	void create_Skyrmion();
	void create_SpinSpiral();
	// Transitions
	void homogeneousTransitionPressed();
	// Debug?
	void print_Energies_to_console();
};

#endif