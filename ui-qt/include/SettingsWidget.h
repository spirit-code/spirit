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
	void ReadExchange();
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
	QRegularExpressionValidator * number_vali;
	QRegularExpressionValidator * number_vali_unsigned;

private slots:
	// Parameters
	void set_parameters();
	// Configurations
	void set_hamiltonian_iso();
	void set_hamiltonian_aniso();
	// Hamiltonian (isotropic)
	void set_extB(std::shared_ptr<Data::Spin_System> ss);
	void set_exchange(std::shared_ptr<Data::Spin_System> ss);
	void set_dt(std::shared_ptr<Data::Spin_System> ss);
	void set_dmi(std::shared_ptr<Data::Spin_System> ss);
	void set_aniso(std::shared_ptr<Data::Spin_System> ss);
	void set_spc(std::shared_ptr<Data::Spin_System> ss);
	void set_bqe(std::shared_ptr<Data::Spin_System> ss);
	void set_fourspin(std::shared_ptr<Data::Spin_System> ss);
	void set_temper(std::shared_ptr<Data::Spin_System> ss);
	// Hamiltonian (anisotropic)
	void set_extB_Anisotropic(std::shared_ptr<Data::Spin_System> ss);
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
	// Parameters
	void set_damping(std::shared_ptr<Data::Spin_System> ss);
	void set_mu_spin(std::shared_ptr<Data::Spin_System> ss);
	void set_spring_constant();
	void set_climbing_falling();
	void set_periodical(std::shared_ptr<Data::Spin_System> ss);
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