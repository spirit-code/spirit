#pragma once
#ifndef HamiltonianAnisotropicWidget_H
#define HamiltonianAnisotropicWidget_H

#include <QWidget>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_HamiltonianAnisotropicWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class HamiltonianAnisotropicWidget : public QWidget, private Ui::HamiltonianAnisotropicWidget
{
    Q_OBJECT

public:
	HamiltonianAnisotropicWidget(std::shared_ptr<State> state, SpinWidget * spinWidget);
	void updateData();

private slots:
	void set_hamiltonian_aniso_bc();
	void set_hamiltonian_aniso_mu_s();
	void set_hamiltonian_aniso_field();
	void set_hamiltonian_aniso_ani();


private:
	void Load_Hamiltonian_Anisotropic_Contents();
	void Setup_Input_Validators();
	void Setup_Hamiltonian_Anisotropic_Slots();

	std::shared_ptr<State> state;
	SpinWidget * spinWidget;
	//SettingsWidget * settingsWidget;

	// Last used configuration
	std::string last_configuration;

	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif