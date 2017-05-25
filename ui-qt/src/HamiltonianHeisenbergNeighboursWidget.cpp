#include <QtWidgets>

#include "HamiltonianHeisenbergNeighboursWidget.hpp"

#include <Spirit/System.h>
#include <Spirit/Chain.h>
#include <Spirit/Collection.h>
#include <Spirit/Log.h>
#include <Spirit/Exception.h>
#include <Spirit/Hamiltonian.h>

// Small function for normalization of vectors
template <typename T>
void normalize(T v[3])
{
	T len = 0.0;
	for (int i = 0; i < 3; ++i) len += std::pow(v[i], 2);
	if (len == 0.0) throw Exception_Division_by_zero;
	for (int i = 0; i < 3; ++i) v[i] /= std::sqrt(len);
}

HamiltonianHeisenbergNeighboursWidget::HamiltonianHeisenbergNeighboursWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
{
	this->state = state;
	this->spinWidget = spinWidget;

	// Setup User Interface
    this->setupUi(this);

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

	// Load variables from State
	this->updateData();

	// Connect signals and slots
	this->Setup_Slots();
}

void HamiltonianHeisenbergNeighboursWidget::updateData()
{
	Load_Contents();
}


void HamiltonianHeisenbergNeighboursWidget::Load_Contents()
{
	float d, dij[100], vd[3], mu_s, jij[100];
	int n_neigh_shells_exchange, n_neigh_shells_dmi;

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
	else this->checkBox_extH->setChecked(false);
	
	// Anisotropy
	Hamiltonian_Get_Anisotropy(state.get(), &d, vd);
	this->lineEdit_aniso->setText(QString::number(d));
	this->lineEdit_anisox->setText(QString::number(vd[0]));
	this->lineEdit_anisoy->setText(QString::number(vd[1]));
	this->lineEdit_anisoz->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_aniso->setChecked(true);
	else this->checkBox_aniso->setChecked(false);

	// Exchange interaction
	Hamiltonian_Get_Exchange(state.get(), &n_neigh_shells_exchange, jij);
	if (n_neigh_shells_exchange > 0) this->checkBox_exchange->setChecked(true);
	else this->checkBox_exchange->setChecked(false);
	this->spinBox_exchange_nshells->setValue(n_neigh_shells_exchange);
	this->set_nshells_exchange();
	for (int i = 0; i < n_neigh_shells_exchange; ++i) this->exchange_shells[i]->setValue(jij[i]);

	// DMI
	Hamiltonian_Get_DMI(state.get(), &n_neigh_shells_dmi, dij);
	if (n_neigh_shells_dmi > 0) this->checkBox_dmi->setChecked(true);
	this->spinBox_dmi_nshells->setValue(n_neigh_shells_dmi);
	this->set_nshells_dmi();
	for (int i = 0; i < n_neigh_shells_dmi; ++i) this->dmi_shells[i]->setValue(dij[i]);

	// DDI
	float ddi_radius;
	Hamiltonian_Get_DDI(state.get(), &ddi_radius);
	if (ddi_radius > 0) this->checkBox_ddi->setChecked(true);
	else this->checkBox_ddi->setChecked(false);
	this->doubleSpinBox_ddi_radius->setValue(ddi_radius);
}


// -----------------------------------------------------------------------------------
// -------------------------- Setters ------------------------------------------------
// -----------------------------------------------------------------------------------


void HamiltonianHeisenbergNeighboursWidget::set_boundary_conditions()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5], dij[5];
		int i;

		// Boundary conditions
		bool boundary_conditions[3];
		boundary_conditions[0] = this->checkBox_iso_periodical_a->isChecked();
		boundary_conditions[1] = this->checkBox_iso_periodical_b->isChecked();
		boundary_conditions[2] = this->checkBox_iso_periodical_c->isChecked();
		Hamiltonian_Set_Boundary_Conditions(state.get(), boundary_conditions, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
	this->spinWidget->updateBoundingBoxIndicators();
}

void HamiltonianHeisenbergNeighboursWidget::set_mu_s()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5], dij[5];
		int i;

		// mu_s
		float mu_s = lineEdit_muSpin->text().toFloat();
		Hamiltonian_Set_mu_s(state.get(), mu_s, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_external_field()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5], dij[5];
		int i;

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
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_anisotropy()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5], dij[5];
		int i;

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
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_nshells_exchange()
{
	// The desired number of shells
	int n_shells = this->spinBox_exchange_nshells->value();
	// The current number of shells
	int n_shells_current = this->exchange_shells.size();
	// If reduction: remove widgets and set exchange
	if (n_shells < n_shells_current)
	{
		for (int n = n_shells_current; n > n_shells; --n)
		{
			this->exchange_shells.back()->close();
			this->exchange_shells.pop_back();
		}
		this->set_exchange();
	}
	// If increase: add widgets, connect to slots and do nothing
	else
	{
		for (int n = n_shells_current; n < n_shells; ++n)
		{
			auto x = new QDoubleSpinBox();
			x->setRange(-1000, 1000);
			this->exchange_shells.push_back(x);
			this->gridLayout_exchange->addWidget(x);
			connect(x, SIGNAL(editingFinished()), this, SLOT(set_exchange()));
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_exchange()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3], jij[5], dij[5];
		int i;

		if (this->checkBox_exchange->isChecked())
		{
			int n_shells = this->exchange_shells.size();
			std::vector<float> Jij(n_shells);
			for (int i = 0; i < n_shells; ++i) Jij[i] = this->exchange_shells[i]->value();
			Hamiltonian_Set_Exchange(state.get(), n_shells, Jij.data(), idx_image, idx_chain);
		}
		else
			Hamiltonian_Set_Exchange(state.get(), 0, nullptr, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_nshells_dmi()
{
	// The desired number of shells
	int n_shells = this->spinBox_dmi_nshells->value();
	// The current number of shells
	int n_shells_current = this->dmi_shells.size();
	// If reduction remove widgets and set dmi
	if (n_shells < n_shells_current)
	{
		for (int n = n_shells_current; n > n_shells; --n)
		{
			this->dmi_shells.back()->close();
			this->dmi_shells.pop_back();
		}
		this->set_dmi();
	}
	// If increase add widgets, connect to slots and do nothing
	else
	{
		for (int n = n_shells_current; n < n_shells; ++n)
		{
			auto x = new QDoubleSpinBox();
			x->setRange(-1000, 1000);
			this->dmi_shells.push_back(x);
			this->gridLayout_dmi->addWidget(x);
			connect(x, SIGNAL(editingFinished()), this, SLOT(set_dmi()));
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_dmi()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		if (this->checkBox_dmi->isChecked())
		{
			int n_shells = this->dmi_shells.size();
			std::vector<float> Dij(n_shells);
			for (int i = 0; i < n_shells; ++i) Dij[i] = this->dmi_shells[i]->value();
			Hamiltonian_Set_DMI(state.get(), n_shells, Dij.data(), idx_image, idx_chain);
		}
		else
			Hamiltonian_Set_DMI(state.get(), 0, nullptr, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

void HamiltonianHeisenbergNeighboursWidget::set_ddi()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		if (this->checkBox_ddi->isChecked())
			Hamiltonian_Set_DDI(state.get(), this->doubleSpinBox_ddi_radius->value(), idx_image, idx_chain);
		else
			Hamiltonian_Set_DDI(state.get(), 0, idx_image, idx_chain);
	};

	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (int ichain = 0; ichain<Collection_Get_NOC(state.get()); ++ichain)
		{
			for (int img = 0; img<Chain_Get_NOI(state.get(), ichain); ++img)
			{
				apply(img, ichain);
			}
		}
	}
}

// -----------------------------------------------------------------------------------
// --------------------------------- Setup -------------------------------------------
// -----------------------------------------------------------------------------------

void HamiltonianHeisenbergNeighboursWidget::Setup_Input_Validators()
{
	//		mu_s
	this->lineEdit_muSpin->setValidator(this->number_validator);
	//		external field
	this->lineEdit_extH->setValidator(this->number_validator);
	this->lineEdit_extHx->setValidator(this->number_validator);
	this->lineEdit_extHy->setValidator(this->number_validator);
	this->lineEdit_extHz->setValidator(this->number_validator);
	//		anisotropy
	this->lineEdit_aniso->setValidator(this->number_validator);
	this->lineEdit_anisox->setValidator(this->number_validator);
	this->lineEdit_anisoy->setValidator(this->number_validator);
	this->lineEdit_anisoz->setValidator(this->number_validator);
	//		BQE
	this->lineEdit_bqe->setValidator(this->number_validator);
	//		FSC
	this->lineEdit_fourspin->setValidator(this->number_validator);
}



void HamiltonianHeisenbergNeighboursWidget::Setup_Slots()
{
	// Boundary conditions
	connect(this->checkBox_iso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
	connect(this->checkBox_iso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
	connect(this->checkBox_iso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
	// mu_s
	connect(this->lineEdit_muSpin, SIGNAL(returnPressed()), this, SLOT(set_mu_s()));
	// External Magnetic Field
	connect(this->checkBox_extH, SIGNAL(stateChanged(int)), this, SLOT(set_external_field()));
	connect(this->lineEdit_extH, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHx, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHy, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHz, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	// Anisotropy
	connect(this->checkBox_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_aniso, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_anisox, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_anisoy, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_anisoz, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	// Exchange
	connect(this->checkBox_exchange, SIGNAL(stateChanged(int)), this, SLOT(set_exchange()));
	connect(this->spinBox_exchange_nshells, SIGNAL(editingFinished()), this, SLOT(set_nshells_exchange()));
	// DMI
	connect(this->checkBox_dmi, SIGNAL(stateChanged(int)), this, SLOT(set_dmi()));
	connect(this->spinBox_dmi_nshells, SIGNAL(editingFinished()), this, SLOT(set_nshells_dmi()));
	// DDI
	connect(this->checkBox_ddi, SIGNAL(stateChanged(int)), this, SLOT(set_ddi()));
	connect(this->doubleSpinBox_ddi_radius, SIGNAL(editingFinished()), this, SLOT(set_ddi()));
}