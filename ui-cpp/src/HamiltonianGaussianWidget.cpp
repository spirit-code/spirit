#include <QtWidgets>

#include "HamiltonianGaussianWidget.hpp"

#include <Spirit/Log.h>

HamiltonianGaussianWidget::HamiltonianGaussianWidget(std::shared_ptr<State> state)
{
    this->state = state;

    // Setup User Interface
    this->setupUi(this);

    // Load variables from State
    this->updateData();

    connect(this->pushButton_changeHamiltonian, SIGNAL(clicked()), this, SLOT(clicked_change_hamiltonian()));
}

void HamiltonianGaussianWidget::updateData()
{
}

void HamiltonianGaussianWidget::clicked_change_hamiltonian()
{
    bool ok;
    std::string type_str = QInputDialog::getItem( this, "Select the Hamiltonian to use", "",
        {"Heisenberg", "Micromagnetic", "Gaussian"}, 0, false, &ok ).toStdString();
    if( ok )
    {
        if( type_str == "Heisenberg" )
            emit hamiltonianChanged(Hamiltonian_Heisenberg);
        else if( type_str == "Micromagnetic" )
            emit hamiltonianChanged(Hamiltonian_Micromagnetic);
        else if( type_str == "Gaussian" )
            emit hamiltonianChanged(Hamiltonian_Gaussian);
    }
}