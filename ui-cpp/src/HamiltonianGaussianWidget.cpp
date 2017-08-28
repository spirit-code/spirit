#include <QtWidgets>

#include "HamiltonianGaussianWidget.hpp"

#include <Spirit/Log.h>
#include <Spirit/Exception.h>
#include <Spirit/Hamiltonian.h>

HamiltonianGaussianWidget::HamiltonianGaussianWidget(std::shared_ptr<State> state)
{
	this->state = state;

	// Setup User Interface
    this->setupUi(this);

	// Load variables from State
	this->updateData();
}

void HamiltonianGaussianWidget::updateData()
{
}