#pragma once
#ifndef HAMILTONIAN_HEISENBERG_PAIRS_WIDGET_H
#define HAMILTONIAN_HEISENBERG_PAIRS_WIDGET_H

#include <QWidget>

#include <vector>
#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_HamiltonianHeisenbergPairsWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class HamiltonianHeisenbergPairsWidget : public QWidget, private Ui::HamiltonianHeisenbergPairsWidget
{
    Q_OBJECT

public:
	HamiltonianHeisenbergPairsWidget(std::shared_ptr<State> state, SpinWidget * spinWidget);
	void updateData();

private slots:
	void set_boundary_conditions();
	void set_mu_s();
	void set_external_field();
	void set_anisotropy();
	void set_nshells_exchange();
	void set_exchange();
	void set_nshells_dmi();
	void set_dmi();
	void set_ddi();
	void set_pairs_from_file();
	void set_pairs_from_text();


private:
	void Load_Contents();
	void Setup_Input_Validators();
	void Setup_Slots();

	std::shared_ptr<State> state;
	SpinWidget * spinWidget;

	// Spinboxes for interaction shells
	std::vector<QDoubleSpinBox *> exchange_shells;
	std::vector<QDoubleSpinBox *> dmi_shells;

	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif