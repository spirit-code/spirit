#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QtWidgets>

#include <memory>

#include "Spin_System.h"
#include "Spin_System_Chain.h"


#include "ui_SettingsWidget.h"

class SettingsWidget : public QWidget, private Ui::SettingsWidget
{
    Q_OBJECT


private:
	// Setup Input Validators
	void Setup_Input_Validators();
	// Setup Slots
	void Setup_Configurations_Slots();
	void Setup_Transitions_Slots();
	void Setup_Hamiltonian_Isotropic_Slots();
	void Setup_Hamiltonian_Anisotropic_Slots();
	void Setup_Parameters_Slots();
	// Load a set of parameters from the spin systems
	void Load_Hamiltonian_Isotropic_Contents();
	void Load_Hamiltonian_Anisotropic_Contents();
	void Load_Parameters_Contents();
	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;

private slots:
	// Parameters
	void set_parameters();
	// Configurations
	void set_hamiltonian_iso();
	void set_hamiltonian_aniso();
	// Configurartions
	void randomPressed();
	void domainWallPressed();
	void plusZ();
	void minusZ();
	void greaterLesserToggle();
	void create_Skyrmion();
	void create_SpinSpiral();
	// Transitions
	void homogeneousTransitionPressed();
	// Debug?
	void print_Energies_to_console();

public:
	SettingsWidget(std::shared_ptr<Data::Spin_System_Chain> s_i);
	void update();
	void SelectTab(int index);

	std::shared_ptr<Data::Spin_System> s;
	std::shared_ptr<Data::Spin_System_Chain> c;
	bool greater;
};

#endif