// #include <QtWidgets>

#include "SettingsWidget.hpp"
#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"

#include "Spirit/Log.h"
#include "Spirit/System.h"
#include "Spirit/Chain.h"
#include "Spirit/Collection.h"
#include "Spirit/Hamiltonian.h"
#include "Spirit/Exception.h"

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
	if (H_name == "Isotropic Heisenberg")
	{
		this->hamiltonianIsotropicWidget = new HamiltonianIsotropicWidget(state);
		this->tab_Settings_Hamiltonian->layout()->addWidget(this->hamiltonianIsotropicWidget);
	}
	else if (H_name == "Anisotropic Heisenberg")
	{
		this->hamiltonianAnisotropicWidget = new HamiltonianAnisotropicWidget(state, spinWidget);
		this->tab_Settings_Hamiltonian->layout()->addWidget(this->hamiltonianAnisotropicWidget);
	}
	else
	{
		this->tabWidget_Settings->removeTab(2);
	}

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
	if (H_name == "Isotropic Heisenberg") this->hamiltonianIsotropicWidget->updateData();
	else if (H_name == "Anisotropic Heisenberg") this->hamiltonianAnisotropicWidget->updateData();
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