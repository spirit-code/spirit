#pragma once
#ifndef SPIRIT_GEOMETRYWIDGET_HPP
#define SPIRIT_GEOMETRYWIDGET_HPP

#include "ui_GeometryWidget.h"

#include "SpinWidget.hpp"

#include <QRegularExpressionValidator>
#include <QtWidgets/QWidget>

#include <memory>
#include <thread>

struct State;

class GeometryWidget : public QWidget, private Ui::GeometryWidget
{
    Q_OBJECT

public:
    GeometryWidget( std::shared_ptr<State> state, SpinWidget * spinWidget );
    void updateData();

signals:
    void updateNeeded();

private slots:
    void setNCells();

private:
    void Setup_Input_Validators();
    void Setup_Slots();

    std::shared_ptr<State> state;
    SpinWidget * spinWidget;

    // Validator for Input into lineEdits
    QRegularExpressionValidator * number_validator;
    QRegularExpressionValidator * number_validator_unsigned;
    QRegularExpressionValidator * number_validator_int;
    QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif