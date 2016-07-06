#pragma once
#ifndef DEBUGWIDGET_H
#define DEBUGWIDGET_H

#include <QtWidgets>

#include <memory>
#include "Spin_System.h"
#include "Spin_System_Chain.h"


#include "ui_DebugWidget.h"

class DebugWidget : public QWidget, private Ui::DebugWidget
{
    Q_OBJECT

public:
	DebugWidget(std::shared_ptr<Data::Spin_System_Chain> c);
	void update();

	void LoadFromLog();
	void UpdateFromLog();

	std::shared_ptr<Data::Spin_System> s;
	std::shared_ptr<Data::Spin_System_Chain> c;

private slots:
	void AllPressed();

private:
	int n_log_entries;
};

#endif