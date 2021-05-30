#include "HamiltonianGaussianWidget.hpp"

#include <QtWidgets>

#include <Spirit/Hamiltonian.h>
#include <Spirit/Log.h>

HamiltonianGaussianWidget::HamiltonianGaussianWidget( std::shared_ptr<State> state )
{
    this->state = state;

    // Setup User Interface
    this->setupUi( this );

    // Load variables from State
    this->updateData();
}

void HamiltonianGaussianWidget::updateData() {}