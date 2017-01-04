#pragma once
#ifndef CONTROLWIDGET_H
#define CONTROLWIDGET_H

#include <QWidget>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "SettingsWidget.hpp"

#include "ui_ControlWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class ControlWidget : public QWidget, private Ui::ControlWidget
{
    Q_OBJECT

public:
	ControlWidget(std::shared_ptr<State> state, SpinWidget *spinWidget, SettingsWidget *settingsWidget);
	void updateData();

public slots:
	void play_pause();
	void stop_all();
	void stop_current();
	void next_image();
	void prev_image();
	void cut_image();
	void paste_image(std::string where="current");
	void delete_image();

private slots:
    void resetPressed();
    void xPressed();
    void yPressed();
    void zPressed();
	void save_EPressed();

private:
	void readSettings();
	void writeSettings();
	// State
	std::shared_ptr<State> state;
	// Spin Widget
	SpinWidget *spinWidget;
	// Settings Widget
	SettingsWidget *settingsWidget;
	
	// Method vectors
	std::vector<std::thread> threads_llg;
	std::vector<std::thread> threads_gneb;
	std::thread thread_mmf;

	// ...
	void save_Energies();

protected:
	void closeEvent(QCloseEvent *event);
};

#endif