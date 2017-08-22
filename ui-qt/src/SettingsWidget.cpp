// #include <QtWidgets>

#include "SettingsWidget.hpp"
#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"

#include "Spirit/Hamiltonian.h"

#include <iostream>
#include <memory>


SettingsWidget::SettingsWidget(std::shared_ptr<State> state, SpinWidget *spinWidget)
{
	this->state = state;
    _spinWidget = spinWidget;

	// Setup User Interface
	this->setupUi(this);

	// Configurations
	this->configurationsWidget = new ConfigurationsWidget(state, spinWidget);
	this->tab_Settings_Configurations->layout()->addWidget(this->configurationsWidget);

	// Parameters
	this->parametersWidget = new ParametersWidget(state);
	this->tab_Settings_Parameters->layout()->addWidget(this->parametersWidget);

	// Hamiltonian
	std::string H_name = Hamiltonian_Get_Name(state.get());
	if (H_name == "Heisenberg (Neighbours)")
	{
		this->hamiltonianHeisenbergNeighboursWidget = new HamiltonianHeisenbergNeighboursWidget(state, spinWidget);
		this->tab_Settings_Hamiltonian->layout()->addWidget(this->hamiltonianHeisenbergNeighboursWidget);
	}
	else if (H_name == "Heisenberg (Pairs)")
	{
		this->hamiltonianHeisenbergPairsWidget = new HamiltonianHeisenbergPairsWidget(state, spinWidget);
		this->tab_Settings_Hamiltonian->layout()->addWidget(this->hamiltonianHeisenbergPairsWidget);
	}
	else if (H_name == "Gaussian")
	{
		this->hamiltonianGaussianWidget = new HamiltonianGaussianWidget(state);
		this->tab_Settings_Hamiltonian->layout()->addWidget(this->hamiltonianGaussianWidget);
	}
	else
	{
		this->tabWidget_Settings->removeTab(2);
	}

	// Geometry
	this->geometryWidget = new GeometryWidget(state, spinWidget);
	this->tab_Settings_Geometry->layout()->addWidget(this->geometryWidget);
	connect(this->geometryWidget, SIGNAL(updateNeeded()), this, SLOT(updateData()));

	// Visualisation
	this->visualisationSettingsWidget = new VisualisationSettingsWidget(state, spinWidget);
	this->tab_Settings_Visualisation->layout()->addWidget(this->visualisationSettingsWidget);
}

void SettingsWidget::updateData()
{
	// Parameters
	this->parametersWidget->updateData();
	// Hamiltonian
	std::string H_name = Hamiltonian_Get_Name(state.get());
	if (H_name == "Heisenberg (Neighbours)") this->hamiltonianHeisenbergNeighboursWidget->updateData();
	else if (H_name == "Heisenberg (Pairs)") this->hamiltonianHeisenbergPairsWidget->updateData();
	else if (H_name == "Gaussian") this->hamiltonianGaussianWidget->updateData();
	// Geometry
	this->geometryWidget->updateData();
	// Visualisation
	this->visualisationSettingsWidget->updateData();

	// ToDo: Also update Debug etc!
}


void SettingsWidget::SelectTab(int index)
{
	this->tabWidget_Settings->setCurrentIndex(index);
}

void SettingsWidget::incrementNCellStep(int increment)
{
	this->visualisationSettingsWidget->incrementNCellStep(increment);
}

void SettingsWidget::lastConfiguration()
{
	this->configurationsWidget->lastConfiguration();
}
void SettingsWidget::randomPressed()
{
	this->configurationsWidget->randomPressed();
}
void SettingsWidget::configurationAddNoise()
{
	this->configurationsWidget->configurationAddNoise();
}