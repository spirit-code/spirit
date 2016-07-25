#pragma once
#ifndef DEBUGWIDGET_H
#define DEBUGWIDGET_H

#include <QtWidgets>

#include <memory>
#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Interface_State.h"

#include "ui_DebugWidget.h"

class DebugWidget : public QWidget, private Ui::DebugWidget
{
    Q_OBJECT

public:
	DebugWidget(std::shared_ptr<State> state);
	void update();

	void LoadFromLog();
	void UpdateFromLog();

	std::shared_ptr<State> state;

private slots:
	void AllPressed();

private:
	int n_log_entries;
};

#endif