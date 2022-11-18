#pragma once
#ifndef SPIRIT_HAMILTONIANGAUSSIANWIDGET_HPP
#define SPIRIT_HAMILTONIANGAUSSIANWIDGET_HPP

#include "ui_HamiltonianGaussianWidget.h"

#include "IsosurfaceWidget.hpp"
#include "SpinWidget.hpp"

#include <QtWidgets/QWidget>

#include <memory>
#include <thread>

#include <Spirit/Hamiltonian.h>

#include "IsosurfaceWidget.hpp"
#include "SpinWidget.hpp"

struct State;

class HamiltonianGaussianWidget : public QWidget, private Ui::HamiltonianGaussianWidget
{
    Q_OBJECT

public:
    HamiltonianGaussianWidget( std::shared_ptr<State> state );
    void updateData();

    std::shared_ptr<State> state;

signals:
    void hamiltonianChanged( Hamiltonian_Type newType );

private slots:
    void clicked_change_hamiltonian();
};

#endif