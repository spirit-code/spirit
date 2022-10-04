#pragma once
#ifndef SPIRIT_INFOWIDGET_HPP
#define SPIRIT_INFOWIDGET_HPP

#include "ui_InfoWidget.h"

#include <QtWidgets/QWidget>

#include <memory>

struct State;
class SpinWidget;
class ControlWidget;

class InfoWidget : public QWidget, private Ui::InfoWidget
{
    Q_OBJECT

public:
    InfoWidget( std::shared_ptr<State> state, SpinWidget * spinWidget, ControlWidget * controlWidget );

private:
    void updateData();

    std::shared_ptr<State> state;
    SpinWidget * spinWidget;
    ControlWidget * controlWidget;
    QTimer * m_timer;
};

#endif