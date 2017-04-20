#pragma once
#ifndef ParametersWidget_H
#define ParametersWidget_H

#include <QWidget>

#include <memory>
#include <thread>

//#include "SpinWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_ParametersWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class ParametersWidget : public QWidget, private Ui::ParametersWidget
{
    Q_OBJECT

public:
	ParametersWidget(std::shared_ptr<State> state);
	void updateData();

private slots:
	void set_parameters();
   
private:
	void Setup_Input_Validators();
	void Setup_Parameters_Slots();
	void Load_Parameters_Contents();

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