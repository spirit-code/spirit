#pragma once
#ifndef HAMILTONIANGAUSSIANWIDGET_H
#define HAMILTONIANGAUSSIANWIDGET_H

#include <QtWidgets/QWidget>

#include <memory>
#include <thread>

#include <Spirit/Hamiltonian.h>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_HamiltonianGaussianWidget.h"

struct State;

class HamiltonianGaussianWidget : public QWidget, private Ui::HamiltonianGaussianWidget
{
    Q_OBJECT

public:
    HamiltonianGaussianWidget(std::shared_ptr<State> state);
    void updateData();

    std::shared_ptr<State> state;

signals:
    void hamiltonianChanged(Hamiltonian_Type newType);

private slots:
    void clicked_change_hamiltonian();
};

#endif