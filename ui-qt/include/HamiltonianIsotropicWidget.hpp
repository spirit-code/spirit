#pragma once
#ifndef HAMILTONIANISOTROPICWIDGET_H
#define HAMILTONIANISOTROPICWIDGET_H

#include <QWidget>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_HamiltonianIsotropicWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class HamiltonianIsotropicWidget : public QWidget, private Ui::HamiltonianIsotropicWidget
{
    Q_OBJECT

public:
	HamiltonianIsotropicWidget(std::shared_ptr<State> state);
	void updateData();

private slots:
	void set_hamiltonian_iso();

private:
	void Load_Hamiltonian_Isotropic_Contents();
	void Setup_Input_Validators();
	void Setup_Hamiltonian_Isotropic_Slots();

	std::shared_ptr<State> state;
	//SpinWidget * spinWidget;
	//SettingsWidget * settingsWidget;

	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif