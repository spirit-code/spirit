// #include <QtWidgets>

#include "SettingsWidget.hpp"
#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
#include "Spirit/Transitions.h"
#include "Spirit/Log.h"
#include "Spirit/System.h"
#include "Spirit/Chain.h"
#include "Spirit/Collection.h"
#include "Spirit/Hamiltonian.h"
#include "Spirit/Parameters.h"
#include "Spirit/Exception.h"

#include <iostream>
#include <memory>

// Small function for normalization of vectors
template <typename T>
void normalize(T v[3])
{
	T len = 0.0;
	for (int i = 0; i < 3; ++i) len += std::pow(v[i], 2);
	if (len == 0.0) throw Exception_Division_by_zero;
	for (int i = 0; i < 3; ++i) v[i] /= std::sqrt(len);
}

SettingsWidget::SettingsWidget(std::shared_ptr<State> state, SpinWidget *spinWidget)
{
	this->state = state;
    _spinWidget = spinWidget;

	// Setup User Interface
	this->setupUi(this);

	this->configurationsWidget = new ConfigurationsWidget(state, spinWidget);
	this->parametersWidget = new ParametersWidget(state);
	this->visualisationSettingsWidget = new VisualisationSettingsWidget(state, spinWidget);
	this->tab_Settings_Configurations->layout()->addWidget(this->configurationsWidget);
	this->tab_Settings_Parameters->layout()->addWidget(this->parametersWidget);
	this->tab_Settings_Visualisation->layout()->addWidget(this->visualisationSettingsWidget);
	
	// We use a regular expression (regex) to filter the input into the lineEdits
	QRegularExpression re("[+|-]?[\\d]*[\\.]?[\\d]*");
	this->number_validator = new QRegularExpressionValidator(re);
	QRegularExpression re2("[\\d]*[\\.]?[\\d]*");
	this->number_validator_unsigned = new QRegularExpressionValidator(re2);
	QRegularExpression re3("[+|-]?[\\d]*");
	this->number_validator_int = new QRegularExpressionValidator(re3);
	QRegularExpression re4("[\\d]*");
	this->number_validator_int_unsigned = new QRegularExpressionValidator(re4);
	// Setup the validators for the various input fields
	this->Setup_Input_Validators();

	// Setup Configurations Tab
	//this->greater = true;
	//this->pushButton_GreaterLesser->setText("Greater");

	// Setup Transitions Tab
	this->lineEdit_Transition_Homogeneous_Last->setText(QString::number(Chain_Get_NOI(this->state.get())));

	// Setup Interactions Tab
	std::string H_name = Hamiltonian_Get_Name(state.get());
	if (H_name == "Isotropic Heisenberg")
	{
		this->tabWidget_Settings->removeTab(3);
		this->Load_Hamiltonian_Isotropic_Contents();
	}
	else if (H_name == "Anisotropic Heisenberg")
	{
		this->tabWidget_Settings->removeTab(2);
		this->Load_Hamiltonian_Anisotropic_Contents();
	}
	else
	{
		this->tabWidget_Settings->removeTab(2);
		this->tabWidget_Settings->removeTab(2);
	}


	// Connect slots
	this->Setup_Transitions_Slots();
	this->Setup_Hamiltonian_Isotropic_Slots();
	this->Setup_Hamiltonian_Anisotropic_Slots();
}

void SettingsWidget::updateData()
{
	// Load Hamiltonian Contents
	std::string H_name = Hamiltonian_Get_Name(state.get());
	if (H_name == "Isotropic Heisenberg") this->Load_Hamiltonian_Isotropic_Contents();
	else if (H_name == "Anisotropic Heisenberg") this->Load_Hamiltonian_Anisotropic_Contents();
	// Update Sub-Widgets
	this->parametersWidget->updateData();
	this->visualisationSettingsWidget->updateData();
	// ToDo: Also update Debug etc!
}





// -----------------------------------------------------------------------------------
// --------------------- Transitions -------------------------------------------------
// -----------------------------------------------------------------------------------

void SettingsWidget::homogeneousTransitionPressed()
{
	int idx_1 = this->lineEdit_Transition_Homogeneous_First->text().toInt() - 1;
	int idx_2 = this->lineEdit_Transition_Homogeneous_Last->text().toInt() - 1;

	int noi = Chain_Get_NOI(this->state.get());

	// Check the validity of the indices
	if (idx_1 < 0 || idx_1 >= noi)
	{
		Log_Send(state.get(), Log_Level_Error, Log_Sender_UI, "First index for homogeneous transition is invalid! setting to 1...");
		this->lineEdit_Transition_Homogeneous_First->setText(QString::number(1));
		return;
	}
	if (idx_2 < 0 || idx_2 >= noi)
	{
		Log_Send(state.get(), Log_Level_Error, Log_Sender_UI, "Second index for homogeneous transition is invalid! setting to NOI...");
		this->lineEdit_Transition_Homogeneous_Last->setText(QString::number(noi));
		return;
	}
	if (idx_1 == idx_2)
	{
		Log_Send(state.get(), Log_Level_Error, Log_Sender_UI, "Indices are equal in homogeneous transition! Aborting...");
		return;
	}
	if (idx_2 < idx_1)
	{
		Log_Send(state.get(), Log_Level_Error, Log_Sender_UI, "Index 2 is smaller than index 1 in homogeneous transition! Aborting...");
		return;
	}

	// Do the transition
	Transition_Homogeneous(this->state.get(), idx_1, idx_2);

	// Add Noise
	if (this->checkBox_Transition_Noise->isChecked())
	{
		float temperature = lineEdit_Transition_Noise->text().toFloat();
		Transition_Add_Noise_Temperature(this->state.get(), temperature, idx_1, idx_2);
	}

	// Update
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}


// -----------------------------------------------------------------------------------
// --------------------- Load Contents -----------------------------------------------
// -----------------------------------------------------------------------------------


void SettingsWidget::Load_Hamiltonian_Isotropic_Contents()
{
	float d, vd[3], mu_s, jij[5];
	int n_neigh_shells;

	// Boundary conditions
	bool boundary_conditions[3];
	Hamiltonian_Get_Boundary_Conditions(state.get(), boundary_conditions);
	this->checkBox_iso_periodical_a->setChecked(boundary_conditions[0]);
	this->checkBox_iso_periodical_b->setChecked(boundary_conditions[1]);
	this->checkBox_iso_periodical_c->setChecked(boundary_conditions[2]);

	// mu_s
	Hamiltonian_Get_mu_s(state.get(), &mu_s);
	this->lineEdit_muSpin->setText(QString::number(mu_s));

	// External magnetic field
	Hamiltonian_Get_Field(state.get(), &d, vd);
	this->lineEdit_extH->setText(QString::number(d));
	this->lineEdit_extHx->setText(QString::number(vd[0]));
	this->lineEdit_extHy->setText(QString::number(vd[1]));
	this->lineEdit_extHz->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_extH->setChecked(true);
	
	// Exchange interaction
	Hamiltonian_Get_Exchange(state.get(), &n_neigh_shells, jij);
	if (n_neigh_shells > 0) {
		lineEdit_exchange1->setText(QString::number(jij[0]));
		lineEdit_exchange1->setEnabled(true);
	}
	else { lineEdit_exchange1->hide(); }
	if (n_neigh_shells > 1) {
		lineEdit_exchange2->setText(QString::number(jij[1]));
		lineEdit_exchange2->setEnabled(true);
	}
	else { lineEdit_exchange2->hide(); }
	if (n_neigh_shells > 2) {
		lineEdit_exchange3->setText(QString::number(jij[2]));
		lineEdit_exchange3->setEnabled(true);
	}
	else { lineEdit_exchange3->hide(); }
	if (n_neigh_shells > 3) {
		lineEdit_exchange4->setText(QString::number(jij[3]));
		lineEdit_exchange4->setEnabled(true);
	}
	else { lineEdit_exchange4->hide(); }
	if (n_neigh_shells > 4) {
		lineEdit_exchange5->setText(QString::number(jij[4]));
		lineEdit_exchange5->setEnabled(true);
	}
	else { lineEdit_exchange5->hide(); }
	checkBox_exchange->setChecked(true);

	// DMI
	Hamiltonian_Get_DMI(state.get(), &d);
	this->lineEdit_dmi->setText(QString::number(d));
	if (d > 0.0) this->checkBox_dmi->setChecked(true);

	// Anisotropy
	Hamiltonian_Get_Anisotropy(state.get(), &d, vd);
	this->lineEdit_aniso->setText(QString::number(d));
	this->lineEdit_anisox->setText(QString::number(vd[0]));
	this->lineEdit_anisoy->setText(QString::number(vd[1]));
	this->lineEdit_anisoz->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_aniso->setChecked(true);

	// BQE
	Hamiltonian_Get_BQE(state.get(), &d);
	this->lineEdit_bqe->setText(QString::number(d));
	if (d > 0.0) this->checkBox_bqe->setChecked(true);

	// FourSpin
	Hamiltonian_Get_FSC(state.get(), &d);
	this->lineEdit_fourspin->setText(QString::number(d));
	if (d > 0.0) this->checkBox_fourspin->setChecked(true);
}



void SettingsWidget::Load_Hamiltonian_Anisotropic_Contents()
{
	float d, vd[3], mu_s;

	// Boundary conditions
	bool boundary_conditions[3];
	Hamiltonian_Get_Boundary_Conditions(state.get(), boundary_conditions);
	this->checkBox_aniso_periodical_a->setChecked(boundary_conditions[0]);
	this->checkBox_aniso_periodical_b->setChecked(boundary_conditions[1]);
	this->checkBox_aniso_periodical_c->setChecked(boundary_conditions[2]);

	// mu_s
	Hamiltonian_Get_mu_s(state.get(), &mu_s);
	this->lineEdit_muSpin_aniso->setText(QString::number(mu_s));

	// External magnetic field
	Hamiltonian_Get_Field(state.get(), &d, vd);
	this->lineEdit_extH_aniso->setText(QString::number(d));
	this->lineEdit_extHx_aniso->setText(QString::number(vd[0]));
	this->lineEdit_extHy_aniso->setText(QString::number(vd[1]));
	this->lineEdit_extHz_aniso->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_extH_aniso->setChecked(true);

	// Anisotropy
	Hamiltonian_Get_Anisotropy(state.get(), &d, vd);
	this->lineEdit_ani_aniso->setText(QString::number(d));
	this->lineEdit_anix_aniso->setText(QString::number(vd[0]));
	this->lineEdit_aniy_aniso->setText(QString::number(vd[1]));
	this->lineEdit_aniz_aniso->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_ani_aniso->setChecked(true);
}



// -----------------------------------------------------------------------------------
// --------------------- Setters for Hamiltonians and Parameters ---------------------
// -----------------------------------------------------------------------------------



void SettingsWidget::set_hamiltonian_iso()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5];
		int i;

		// Boundary conditions
		bool boundary_conditions[3];
		boundary_conditions[0] = this->checkBox_iso_periodical_a->isChecked();
		boundary_conditions[1] = this->checkBox_iso_periodical_b->isChecked();
		boundary_conditions[2] = this->checkBox_iso_periodical_c->isChecked();
		Hamiltonian_Set_Boundary_Conditions(state.get(), boundary_conditions, idx_image, idx_chain);
		
		// mu_s
		float mu_s = lineEdit_muSpin->text().toFloat();
		Hamiltonian_Set_mu_s(state.get(), mu_s, idx_image, idx_chain);

		// External magnetic field
		//		magnitude
		if (this->checkBox_extH->isChecked())
			d = this->lineEdit_extH->text().toFloat();
		else d = 0.0;
		//		normal
		vd[0] = lineEdit_extHx->text().toFloat();
		vd[1] = lineEdit_extHy->text().toFloat();
		vd[2] = lineEdit_extHz->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "B_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_extHx->setText(QString::number(0.0));
				lineEdit_extHy->setText(QString::number(0.0));
				lineEdit_extHz->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_Field(state.get(), d, vd, idx_image, idx_chain);

		// Exchange
		i=0;
		if (lineEdit_exchange1->isEnabled()) { jij[0] = lineEdit_exchange1->text().toFloat(); ++i; }
		if (lineEdit_exchange2->isEnabled()) { jij[1] = lineEdit_exchange2->text().toFloat(); ++i; }
		if (lineEdit_exchange3->isEnabled()) { jij[2] = lineEdit_exchange3->text().toFloat(); ++i; }
		if (lineEdit_exchange4->isEnabled()) { jij[3] = lineEdit_exchange4->text().toFloat(); ++i; }
		if (lineEdit_exchange5->isEnabled()) { jij[4] = lineEdit_exchange5->text().toFloat(); ++i; }
		if (!checkBox_exchange->isChecked())
		{
			for (int shell = 0; shell < i; ++shell) {
				jij[shell] = 0.0;
			}
		}
		Hamiltonian_Set_Exchange(state.get(), i, jij, idx_image, idx_chain);
		
		// DMI
		if (this->checkBox_dmi->isChecked()) d = this->lineEdit_dmi->text().toFloat();
		else d = 0.0;
		Hamiltonian_Set_DMI(state.get(), d, idx_image, idx_chain);

		// Anisotropy
		//		magnitude
		if (this->checkBox_aniso->isChecked()) d = this->lineEdit_aniso->text().toFloat();
		else d = 0.0;
		//		normal
		vd[0] = lineEdit_anisox->text().toFloat();
		vd[1] = lineEdit_anisoy->text().toFloat();
		vd[2] = lineEdit_anisoz->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "Aniso_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_anisox->setText(QString::number(0.0));
				lineEdit_anisoy->setText(QString::number(0.0));
				lineEdit_anisoz->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_Anisotropy(state.get(), d, vd, idx_image, idx_chain);

		// BQE
		if (this->checkBox_bqe->isChecked()) d = this->lineEdit_bqe->text().toFloat();
		else d = 0.0;
		Hamiltonian_Set_BQE(state.get(), d, idx_image, idx_chain);

		// FSC
		if (this->checkBox_fourspin->isChecked()) d = this->lineEdit_fourspin->text().toFloat();
		else d = 0.0;
		Hamiltonian_Set_FSC(state.get(), d, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i=0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain=0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void SettingsWidget::set_hamiltonian_aniso_bc()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		// Boundary conditions
		bool boundary_conditions[3];
		boundary_conditions[0] = this->checkBox_aniso_periodical_a->isChecked();
		boundary_conditions[1] = this->checkBox_aniso_periodical_b->isChecked();
		boundary_conditions[2] = this->checkBox_aniso_periodical_c->isChecked();
		Hamiltonian_Set_Boundary_Conditions(state.get(), boundary_conditions, idx_image, idx_chain);
	};
	
	if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i=0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
	{
		for (int ichain=0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
	this->_spinWidget->updateBoundingBoxIndicators();
}

void SettingsWidget::set_hamiltonian_aniso_mu_s()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		// mu_s
		float mu_s = this->lineEdit_muSpin_aniso->text().toFloat();
		Hamiltonian_Set_mu_s(state.get(), mu_s, idx_image, idx_chain);
	};
	
	if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i=0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
	{
		for (int ichain=0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void SettingsWidget::set_hamiltonian_aniso_field()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3];

		// External magnetic field
		//		magnitude
		if (this->checkBox_extH_aniso->isChecked()) d = this->lineEdit_extH_aniso->text().toFloat();
		else d = 0.0;
		//		normal
		vd[0] = lineEdit_extHx_aniso->text().toFloat();
		vd[1] = lineEdit_extHy_aniso->text().toFloat();
		vd[2] = lineEdit_extHz_aniso->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "B_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_extHx_aniso->setText(QString::number(0.0));
				lineEdit_extHy_aniso->setText(QString::number(0.0));
				lineEdit_extHz_aniso->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_Field(state.get(), d, vd, idx_image, idx_chain);
	};
	
	if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i=0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
	{
		for (int ichain=0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void SettingsWidget::set_hamiltonian_aniso_ani()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3];

		// Anisotropy
		//		magnitude
		if (this->checkBox_ani_aniso->isChecked()) d = this->lineEdit_ani_aniso->text().toFloat();
		else d = 0.0;
		//		normal
		vd[0] = lineEdit_anix_aniso->text().toFloat();
		vd[1] = lineEdit_aniy_aniso->text().toFloat();
		vd[2] = lineEdit_aniz_aniso->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "ani_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_anix_aniso->setText(QString::number(0.0));
				lineEdit_aniy_aniso->setText(QString::number(0.0));
				lineEdit_aniz_aniso->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_Anisotropy(state.get(), d, vd, idx_image, idx_chain);
	};
	
	if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i=0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
	{
		for (int ichain=0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}




// -----------------------------------------------------------------------------------
// --------------------- Utilities ---------------------------------------------------
// -----------------------------------------------------------------------------------


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


// -----------------------------------------------------------------------------------
// --------------------- Setup functions for Slots and Validators --------------------
// -----------------------------------------------------------------------------------

void SettingsWidget::Setup_Hamiltonian_Isotropic_Slots()
{
	// Boundary conditions
	connect(this->checkBox_iso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->checkBox_iso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->checkBox_iso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	// mu_s
	connect(this->lineEdit_muSpin, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// External Magnetic Field
	connect(this->checkBox_extH, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_extH, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_extHx, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_extHy, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_extHz, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Exchange
	connect(this->lineEdit_exchange1, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_exchange2, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_exchange3, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_exchange4, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_exchange5, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->checkBox_exchange, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	// DMI
	connect(this->checkBox_dmi, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_dmi, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Anisotropy
	connect(this->checkBox_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_anisox, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_anisoy, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_anisoz, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Biquadratic Exchange
	connect(this->checkBox_bqe, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_bqe, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// FourSpin Interaction
	connect(this->checkBox_fourspin, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_fourspin, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));

}

void SettingsWidget::Setup_Hamiltonian_Anisotropic_Slots()
{
	// Boundary Conditions
	connect(this->checkBox_aniso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_bc()));
	connect(this->checkBox_aniso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_bc()));
	connect(this->checkBox_aniso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_bc()));
	// mu_s
	connect(this->lineEdit_muSpin_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_mu_s()));
	// External Field
	connect(this->checkBox_extH_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_field()));
	connect(this->lineEdit_extH_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_field()));
	connect(this->lineEdit_extHx_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_field()));
	connect(this->lineEdit_extHy_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_field()));
	connect(this->lineEdit_extHz_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_field()));
	// Anisotropy
	connect(this->checkBox_ani_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_ani()));
	connect(this->lineEdit_ani_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_ani()));
	connect(this->lineEdit_anix_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_ani()));
	connect(this->lineEdit_aniy_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_ani()));
	connect(this->lineEdit_aniz_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_ani()));
}



void SettingsWidget::Setup_Transitions_Slots()
{
	// Homogeneous Transition
	connect(this->pushButton_Transition_Homogeneous, SIGNAL(clicked()), this, SLOT(homogeneousTransitionPressed()));
}

void SettingsWidget::Setup_Input_Validators()
{
	// Isotropic Hamiltonian
	//		mu_s
	this->lineEdit_muSpin->setValidator(this->number_validator);
	//		external field
	this->lineEdit_extH->setValidator(this->number_validator);
	this->lineEdit_extHx->setValidator(this->number_validator);
	this->lineEdit_extHy->setValidator(this->number_validator);
	this->lineEdit_extHz->setValidator(this->number_validator);
	//		exchange
	this->lineEdit_exchange1->setValidator(this->number_validator);
	this->lineEdit_exchange2->setValidator(this->number_validator);
	this->lineEdit_exchange3->setValidator(this->number_validator);
	this->lineEdit_exchange4->setValidator(this->number_validator);
	this->lineEdit_exchange5->setValidator(this->number_validator);
	//		DMI
	this->lineEdit_dmi->setValidator(this->number_validator);
	//		anisotropy
	this->lineEdit_aniso->setValidator(this->number_validator);
	this->lineEdit_anisox->setValidator(this->number_validator);
	this->lineEdit_anisoy->setValidator(this->number_validator);
	this->lineEdit_anisoz->setValidator(this->number_validator);
	//		BQE
	this->lineEdit_bqe->setValidator(this->number_validator);
	//		FSC
	this->lineEdit_fourspin->setValidator(this->number_validator);

	// Anisotropic Hamiltonian
	//		mu_s
	this->lineEdit_muSpin_aniso->setValidator(this->number_validator);
	//		external field
	this->lineEdit_extH_aniso->setValidator(this->number_validator);
	this->lineEdit_extHx_aniso->setValidator(this->number_validator);
	this->lineEdit_extHy_aniso->setValidator(this->number_validator);
	this->lineEdit_extHz_aniso->setValidator(this->number_validator);
	//		anisotropy
	this->lineEdit_ani_aniso->setValidator(this->number_validator);
	this->lineEdit_anix_aniso->setValidator(this->number_validator);
	this->lineEdit_aniy_aniso->setValidator(this->number_validator);
	this->lineEdit_aniz_aniso->setValidator(this->number_validator);

	// Transitions
	this->lineEdit_Transition_Noise->setValidator(this->number_validator_unsigned);
	this->lineEdit_Transition_Homogeneous_First->setValidator(this->number_validator_int_unsigned);
	this->lineEdit_Transition_Homogeneous_Last->setValidator(this->number_validator_int_unsigned);
}
