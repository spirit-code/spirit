#pragma once
#ifndef INFOWIDGET_H
#define INFOWIDGET_H

#include <QtWidgets/QWidget>

#include <memory>

#include <VFRendering/IsosurfaceRenderer.hxx>

#include "ui_InfoWidget.h"

struct State;
class SpinWidget;

class InfoWidget : public QWidget, private Ui::InfoWidget
{
    Q_OBJECT

public:
    InfoWidget(std::shared_ptr<State> state, SpinWidget *spinWidget);

private:
    void updateData();

    std::shared_ptr<State> state;
    SpinWidget * spinWidget;
    QTimer * m_timer;
};

#endif