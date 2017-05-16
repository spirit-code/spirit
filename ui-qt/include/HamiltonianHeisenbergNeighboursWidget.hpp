#pragma once
#ifndef HAMILTONIAN_HEISENBERG_NEIGHBOURS_WIDGET_H
#define HAMILTONIAN_HEISENBERG_NEIGHBOURS_WIDGET_H

#include <QWidget>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_HamiltonianHeisenbergNeighboursWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class HamiltonianHeisenbergNeighboursWidget : public QWidget, private Ui::HamiltonianHeisenbergNeighboursWidget
{
    Q_OBJECT

public:
	HamiltonianHeisenbergNeighboursWidget(std::shared_ptr<State> state);
	void updateData();

private slots:
	void set_hamiltonian_iso();

private:
	void Load_Hamiltonian_Heisenberg_Neighbours_Contents();
	void Setup_Input_Validators();
	void Setup_Hamiltonian_Heisenberg_Neighbours_Slots();

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