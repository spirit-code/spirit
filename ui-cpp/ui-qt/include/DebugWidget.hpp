#pragma once
#ifndef SPIRIT_DEBUGWIDGET_HPP
#define SPIRIT_DEBUGWIDGET_HPP

#include "ui_DebugWidget.h"

#include <QWidget>

#include <memory>

struct State;

class DebugWidget : public QWidget, private Ui::DebugWidget
{
    Q_OBJECT

public:
    DebugWidget( std::shared_ptr<State> state );
    void updateData();

    void LoadFromLog();
    void UpdateFromLog();

    std::shared_ptr<State> state;

private slots:
    void AllPressed();

private:
    int n_log_entries;
};

#endif