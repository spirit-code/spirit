#pragma once
#ifndef DEBUGWIDGET_H
#define DEBUGWIDGET_H

#include <QWidget>

#include <memory>

#include "ui_DebugWidget.h"

struct State;

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