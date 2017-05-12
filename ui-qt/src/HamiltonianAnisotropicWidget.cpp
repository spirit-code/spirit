#include <QtWidgets>

#include "HamiltonianAnisotropicWidget.hpp"

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

HamiltonianAnisotropicWidget::HamiltonianAnisotropicWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
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

	// Load variables from SpinWidget and State
	this->updateData();

	// Connect signals and slots
	this->Setup_Hamiltonian_Anisotropic_Slots();
}

void HamiltonianAnisotropicWidget::updateData()
{
	Load_Hamiltonian_Anisotropic_Contents();
}



void HamiltonianAnisotropicWidget::Load_Hamiltonian_Anisotropic_Contents()
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
// -------------------------- Setters ------------------------------------------------
// -----------------------------------------------------------------------------------


void HamiltonianAnisotropicWidget::set_hamiltonian_aniso_bc()
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
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
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

void HamiltonianAnisotropicWidget::set_hamiltonian_aniso_mu_s()
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
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
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

void HamiltonianAnisotropicWidget::set_hamiltonian_aniso_field()
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
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
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

void HamiltonianAnisotropicWidget::set_hamiltonian_aniso_ani()
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
		for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
		{
			apply(i, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
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

void HamiltonianAnisotropicWidget::Setup_Input_Validators()
{
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
}




void HamiltonianAnisotropicWidget::Setup_Hamiltonian_Anisotropic_Slots()
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