#include <QtWidgets>
//#include <QString>
//#include <QtCore>

#include "SettingsWidget.h"

#include "Vectormath.h"
#include "Configurations.h"
#include "Configuration_Chain.h"
#include "Exception.h"
#include <iostream>
#include <memory>

#include"Logging.h"
//extern Utility::LoggingHandler Log;

SettingsWidget::SettingsWidget(std::shared_ptr<Data::Spin_System_Chain> c)
{
	this->c = c;
	this->s = c->images[c->active_image];

	// Setup User Interface
	this->setupUi(this);

	// We use a regular expression (regex) to filter the input into the lineEdits
	QRegularExpression re("[+|-]?[\\d]*[\\.]?[\\d]*");
	number_vali = new QRegularExpressionValidator(re);
	QRegularExpression re2("[\\d]*[\\.]?[\\d]*");
	number_vali_unsigned = new QRegularExpressionValidator(re2);

	// Setup Configurations Tab
	this->greater = true;
	this->pushButton_GreaterLesser->setText("Greater");

	// Setup Transitions Tab
	this->lineEdit_Transition_Homogeneous_Last->setText(QString::number(this->c->noi));

	// Setup Interactions Tab
	if (this->s->is_isotropic)
	{
		this->tabWidget_Settings->removeTab(3);
	}
	else
	{
		this->tabWidget_Settings->removeTab(2);
	}

	// Load information from Spin Systems
	this->update();

	// Connect slots
	this->Setup_Configurations_Slots();
	this->Setup_Transitions_Slots();
	this->Setup_Hamiltonian_Isotropic_Slots();
	this->Setup_Hamiltonian_Anisotropic_Slots();
	this->Setup_Parameters_Slots();
}

void SettingsWidget::update()
{
	this->s = this->c->images[this->c->active_image];
	if (this->s->is_isotropic) this->Load_Hamiltonian_Isotropic_Contents();
	else this->Load_Hamiltonian_Anisotropic_Contents();
	this->Load_Parameters_Contents();
	// ToDo: Also update Debug etc!
}


void SettingsWidget::SelectTab(int index)
{
	this->tabWidget_Settings->setCurrentIndex(index);
}

void SettingsWidget::Setup_Input_Validators()
{
	// Isotropic Hamiltonian
	this->lineEdit_muSpin->setValidator(this->number_vali);

	this->lineEdit_extH->setValidator(this->number_vali);
	this->lineEdit_extHx->setValidator(this->number_vali);
	this->lineEdit_extHy->setValidator(this->number_vali);
	this->lineEdit_extHz->setValidator(this->number_vali);

	this->lineEdit_dmi->setValidator(this->number_vali);

	this->lineEdit_aniso->setValidator(this->number_vali);
	this->lineEdit_anisox->setValidator(this->number_vali);
	this->lineEdit_anisoy->setValidator(this->number_vali);
	this->lineEdit_anisoz->setValidator(this->number_vali);

	this->lineEdit_spin_torque->setValidator(this->number_vali);
	this->lineEdit_spin_torquex->setValidator(this->number_vali);
	this->lineEdit_spin_torquey->setValidator(this->number_vali);
	this->lineEdit_spin_torquez->setValidator(this->number_vali);

	this->lineEdit_bqe->setValidator(this->number_vali);

	this->lineEdit_fourspin->setValidator(this->number_vali);

	this->lineEdit_temper->setValidator(this->number_vali);

	// Anisotropic Hamiltonian
	this->lineEdit_muSpin_aniso->setValidator(this->number_vali);
	this->lineEdit_extH_aniso->setValidator(this->number_vali);
	this->lineEdit_extHx_aniso->setValidator(this->number_vali);
	this->lineEdit_extHy_aniso->setValidator(this->number_vali);
	this->lineEdit_extHz_aniso->setValidator(this->number_vali);

	// Configurations
	this->lineEdit_sky_order->setValidator(this->number_vali);
	this->lineEdit_sky_phase->setValidator(this->number_vali);
	this->lineEdit_sky_rad->setValidator(this->number_vali);
	this->lineEdit_sky_posx->setValidator(this->number_vali);
	this->lineEdit_sky_posy->setValidator(this->number_vali);
	this->lineEdit_sky_posz->setValidator(this->number_vali);

	this->lineEdit_SS_dir_x->setValidator(this->number_vali);
	this->lineEdit_SS_dir_y->setValidator(this->number_vali);
	this->lineEdit_SS_dir_z->setValidator(this->number_vali);
	this->lineEdit_SS_axis_x->setValidator(this->number_vali);
	this->lineEdit_SS_axis_y->setValidator(this->number_vali);
	this->lineEdit_SS_axis_z->setValidator(this->number_vali);
	this->lineEdit_SS_period->setValidator(this->number_vali);

	this->lineEdit_vx->setValidator(this->number_vali);
	this->lineEdit_vy->setValidator(this->number_vali);
	this->lineEdit_vz->setValidator(this->number_vali);
	this->lineEdit_posx->setValidator(this->number_vali);
	this->lineEdit_posy->setValidator(this->number_vali);
	this->lineEdit_posz->setValidator(this->number_vali);

	// Transitions
	this->lineEdit_Transition_Homogeneous_First->setValidator(this->number_vali_unsigned);
	this->lineEdit_Transition_Homogeneous_Last->setValidator(this->number_vali_unsigned);

	// Parameters
	this->lineEdit_Damping->setValidator(this->number_vali);
	this->lineEdit_dt->setValidator(this->number_vali); 

	this->lineEdit_gneb_springconstant->setValidator(this->number_vali);
}

void SettingsWidget::Setup_Transitions_Slots()
{
	// Homogeneous Transition
	connect(this->lineEdit_Transition_Homogeneous_First, SIGNAL(returnPressed()), this, SLOT(homogeneousTransitionPressed()));
	connect(this->lineEdit_Transition_Homogeneous_Last, SIGNAL(returnPressed()), this, SLOT(homogeneousTransitionPressed()));
	connect(this->pushButton_Transition_Homogeneous, SIGNAL(clicked()), this, SLOT(homogeneousTransitionPressed()));
}

void SettingsWidget::homogeneousTransitionPressed()
{
	int idx_1 = this->lineEdit_Transition_Homogeneous_First->text().toInt()-1;
	int idx_2 = this->lineEdit_Transition_Homogeneous_Last->text().toInt()-1;
	
	// Check the validity of the indices
	if (idx_1 < 0 || idx_1 >= this->c->noi)
	{
		Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "First index for homogeneous transition is invalid! setting to 1...");
		this->lineEdit_Transition_Homogeneous_First->setText(QString::number(1));
	}
	if (idx_1 < 0 || idx_1 >= this->c->noi)
	{
		Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "First index for homogeneous transition is invalid! setting to 1...");
		this->lineEdit_Transition_Homogeneous_First->setText(QString::number(1));
	}
	if (idx_1 == idx_2)
	{
		Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "Indices are equal in homogeneous transtion! Aborting...");
		return;
	}
	if (idx_2 < idx_1)
	{
		Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "Index 2 is smaller than index 1 in homogeneous transition! Aborting...");
		return;
	}

	//// Do the transition
	//int n_images = idx_2 - idx_1 + 1;
	//int nos = this->c->images[idx_1]->nos;
	//auto images = std::vector<std::vector<double>&>();// (n_images, std::vector<double>(3 * nos, 0.0));	// [n_images][3nos]
	//for (int i = idx_1; i <= idx_2; ++i)
	//{
	//	images.push_back(this->c->images[i]->spins);
	//}
	Utility::Configuration_Chain::Homogeneous_Rotation(c, idx_1, idx_2);
	//Utility::Configuration_Chain::Homogeneous_Rotation(c, s1->spins, s4->spins);
}

void SettingsWidget::Setup_Parameters_Slots()
{
	// LLG Damping
	connect(this->lineEdit_Damping, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_dt, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// GNEB Spring Constant
	connect(this->lineEdit_gneb_springconstant, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// Normal/Climbing/Falling image radioButtons
	connect(this->radioButton_Normal, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_ClimbingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_FallingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
}

void SettingsWidget::Load_Parameters_Contents()
{
	// LLG Damping
	this->lineEdit_Damping->setText(QString::number(s->llg_parameters->damping));
	this->lineEdit_dt->setText(QString::number(s->llg_parameters->dt));
	// GNEB Spring Constant
	this->lineEdit_gneb_springconstant->setText(QString::number(this->c->gneb_parameters->spring_constant));
	// Normal/Climbing/Falling image radioButtons
	this->radioButton_Normal->setChecked(!(this->c->climbing_image[this->c->active_image] || this->c->falling_image[this->c->active_image]));
	this->radioButton_ClimbingImage->setChecked(this->c->climbing_image[this->c->active_image]);
	this->radioButton_FallingImage->setChecked(this->c->falling_image[this->c->active_image]);
}

void SettingsWidget::Setup_Configurations_Slots()
{
	// Random
	connect(this->pushButton_Random, SIGNAL(clicked()), this, SLOT(randomPressed()));
	// Domain Wall
	connect(this->pushButton_DomainWall, SIGNAL(clicked()), this, SLOT(domainWallPressed()));
	connect(this->pushButton_GreaterLesser, SIGNAL(clicked()), this, SLOT(greaterLesserToggle()));
	// Homogeneous
	connect(this->pushButton_plusZ, SIGNAL(clicked()), this, SLOT(plusZ()));
	connect(this->pushButton_minusZ, SIGNAL(clicked()), this, SLOT(minusZ()));
	// Skyrmion
	connect(this->pushButton_skyrmion, SIGNAL(clicked()), this, SLOT(create_Skyrmion()));
	// Spin Spiral
	connect(this->pushButton_SS, SIGNAL(clicked()), this, SLOT(create_SpinSpiral()));

	// Domain Wall LineEdits
	connect(this->lineEdit_vx, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));
	connect(this->lineEdit_vy, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));
	connect(this->lineEdit_vz, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));
	connect(this->lineEdit_posx, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));
	connect(this->lineEdit_posy, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));
	connect(this->lineEdit_posz, SIGNAL(returnPressed()), this, SLOT(domainWallPressed()));

	// Skyrmion LineEdits
	connect(this->lineEdit_sky_order, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_sky_phase, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_sky_posx, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_sky_posy, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_sky_posz, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_sky_rad, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));

	// SpinSpiral LineEdits
	connect(this->lineEdit_SS_dir_x, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_dir_y, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_dir_z, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_x, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_y, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_z, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_period, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	
}

void SettingsWidget::Load_Hamiltonian_Isotropic_Contents()
{
	//		Read Parameters from Spin System

	// Periodical boundary conditions
	this->checkBox_iso_periodical_a->setChecked(s->hamiltonian->boundary_conditions[0]);
	this->checkBox_iso_periodical_b->setChecked(s->hamiltonian->boundary_conditions[1]);
	this->checkBox_iso_periodical_c->setChecked(s->hamiltonian->boundary_conditions[2]);

	// external field
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		this->lineEdit_muSpin->setText(QString::number(ham->mu_s));
		this->lineEdit_extH->setText(QString::number(ham->external_field_magnitude / Utility::Vectormath::MuB() / ham->mu_s));
		this->lineEdit_extHx->setText(QString::number(ham->external_field_normal[0]));
		this->lineEdit_extHy->setText(QString::number(ham->external_field_normal[1]));
		this->lineEdit_extHz->setText(QString::number(ham->external_field_normal[2]));
		this->checkBox_extH->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	// exchange interaction - connect this shiat
	ReadExchange();

	// DMI
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		this->lineEdit_dmi->setText(QString::number(ham->dij));
		this->checkBox_dmi->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	// anisotropy
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		this->lineEdit_aniso->setText(QString::number(ham->anisotropy_magnitude));
		this->lineEdit_anisox->setText(QString::number(ham->anisotropy_normal[0]));
		this->lineEdit_anisoy->setText(QString::number(ham->anisotropy_normal[1]));
		this->lineEdit_anisoz->setText(QString::number(ham->anisotropy_normal[2]));
		this->checkBox_aniso->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	// spin polarized current (does not really belong to interactions)
	this->lineEdit_spin_torque->setText(QString::number(s->llg_parameters->stt_magnitude));
	this->lineEdit_spin_torquex->setText(QString::number(s->llg_parameters->stt_polarisation_normal[0]));
	this->lineEdit_spin_torquey->setText(QString::number(s->llg_parameters->stt_polarisation_normal[1]));
	this->lineEdit_spin_torquez->setText(QString::number(s->llg_parameters->stt_polarisation_normal[2]));
	this->checkBox_spin_torque->setChecked(true);
	// BQE
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		this->lineEdit_bqe->setText(QString::number(ham->bij));
		this->checkBox_bqe->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	// FourSpin
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		this->lineEdit_fourspin->setText(QString::number(ham->kijkl));
		this->checkBox_fourspin->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	// Temperature (does not really belong to interactions)
	this->lineEdit_temper->setText(QString::number(s->llg_parameters->temperature));
}


void SettingsWidget::Setup_Hamiltonian_Isotropic_Slots()
{
	// Periodical boundary conditions
	connect(this->checkBox_iso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->checkBox_iso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->checkBox_iso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	// External Magnetic Field
	connect(this->lineEdit_muSpin, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
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
	// Spin Torque (does not really belong to interactions)
	connect(this->checkBox_spin_torque, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torque, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquex, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquey, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquez, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Biquadratic Exchange
	connect(this->checkBox_bqe, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_bqe, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// FourSpin Interaction
	connect(this->checkBox_fourspin, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_fourspin, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Temperature (does not really belong to interactions)
	connect(this->lineEdit_temper, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));

}

void SettingsWidget::Load_Hamiltonian_Anisotropic_Contents()
{
	// B-field
	// Anisotropy
	// DMI strength
	// mu_s
	// Periodical boundary conditions
	this->checkBox_aniso_periodical_a->setChecked(s->hamiltonian->boundary_conditions[0]);
	this->checkBox_aniso_periodical_b->setChecked(s->hamiltonian->boundary_conditions[1]);
	this->checkBox_aniso_periodical_c->setChecked(s->hamiltonian->boundary_conditions[2]);
}

void SettingsWidget::Setup_Hamiltonian_Anisotropic_Slots()
{
	connect(this->lineEdit_extH_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso()));
	connect(this->lineEdit_muSpin_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso()));
	connect(this->lineEdit_extHx_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso()));
	connect(this->lineEdit_extHy_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso()));
	connect(this->lineEdit_extHz_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso()));
	connect(this->checkBox_aniso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso()));
	connect(this->checkBox_aniso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso()));
	connect(this->checkBox_aniso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso()));
}



// ------------------ Configurations -------------------------------
void SettingsWidget::randomPressed()
{
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, "button Random");
	Utility::Configurations::Random(*s);
	print_Energies_to_console();
}
void SettingsWidget::minusZ()
{
	Utility::Configurations::MinusZ(*s);
	print_Energies_to_console();
}
void SettingsWidget::plusZ()
{
	Utility::Configurations::PlusZ(*s);
	print_Energies_to_console();
}

void SettingsWidget::greaterLesserToggle()
{
	if (this->greater) {
		this->greater = false;
		this->pushButton_GreaterLesser->setText("Lesser");
	}
	else {
		this->greater = true;
		this->pushButton_GreaterLesser->setText("Greater");
	}
}

void SettingsWidget::create_Skyrmion()
{
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, "button createSkyrmion");
	double speed = lineEdit_sky_order->text().toDouble();
	double phase = lineEdit_sky_phase->text().toDouble();
	bool upDown = checkBox_sky_UpDown->isChecked();
	bool achiral = checkBox_sky_Achiral->isChecked();
	bool rl = checkBox_sky_RL->isChecked();
	bool experimental = checkBox_sky_experimental->isChecked();
	std::vector<double> pos =
	{ 
		lineEdit_sky_posx->text().toDouble() + s->geometry->center[0],
		lineEdit_sky_posy->text().toDouble() + s->geometry->center[1],
		lineEdit_sky_posz->text().toDouble() + s->geometry->center[2]
	};
	double rad = lineEdit_sky_rad->text().toDouble();
	Utility::Configurations::Skyrmion(*s, pos, rad, speed, phase, upDown, achiral, rl, experimental);
	print_Energies_to_console();
}

void SettingsWidget::create_SpinSpiral()
{
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, "button createSpinSpiral");
	double direction[3] = { lineEdit_SS_dir_x->text().toDouble(), lineEdit_SS_dir_y->text().toDouble(), lineEdit_SS_dir_z->text().toDouble() };
	double axis[3] = { lineEdit_SS_axis_x->text().toDouble(), lineEdit_SS_axis_y->text().toDouble(), lineEdit_SS_axis_z->text().toDouble() };
	double period = lineEdit_SS_period->text().toDouble();
	std::string direction_type;
	// And now an ugly workaround because the QT people are too stupid to fix a Bug with QString::toStdString on Windows...
	if (comboBox_SS->currentText() == "Real Lattice") direction_type = "Real Lattice";
	else if (comboBox_SS->currentText() == "Reciprocal Lattice") direction_type = "Reciprocal Lattice";
	else if (comboBox_SS->currentText() == "Real Space") direction_type = "Real Space";
	//Utility::Configurations::SpinSpiral(*s, axis, direction, direction_type, period);
	Utility::Configurations::SpinSpiral(*s, direction_type, direction, axis, period);
	print_Energies_to_console();
}

void SettingsWidget::domainWallPressed()
{
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, "button DomainWall");
	double vec[3] = { lineEdit_vx->text().toDouble(), lineEdit_vy->text().toDouble(), lineEdit_vz->text().toDouble() };
	double pos[3] = { lineEdit_posx->text().toDouble(), lineEdit_posy->text().toDouble(), lineEdit_posz->text().toDouble() };
	Utility::Configurations::DomainWall(*s, pos, vec, this->greater);
	print_Energies_to_console();
}


// ------------------------------ Interactions ----------------------------------------------------------------------------
void SettingsWidget::set_hamiltonian_iso()
{
	if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image")
	{
		this->set_extB(this->s);
		this->set_dmi(this->s);
		this->set_aniso(this->s);
		this->set_spc(this->s);
		this->set_exchange(this->s);
		this->set_bqe(this->s);
		this->set_fourspin(this->s);
		this->set_temper(this->s);
		this->set_mu_spin(this->s);
		this->set_periodical(this->s);
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "Current Image Chain")
	{
		for (auto sys : this->c->images)
		{
			this->set_extB(sys);
			this->set_dmi(sys);
			this->set_aniso(sys);
			this->set_spc(sys);
			this->set_bqe(sys);
			this->set_fourspin(sys);
			this->set_temper(sys);
			this->set_mu_spin(sys);
			this->set_periodical(this->s);
		}
	}
	else if (this->comboBox_Hamiltonian_Iso_ApplyTo->currentText() == "All Images")
	{
		for (auto sys : this->c->images)
		{
			this->set_extB(sys);
			this->set_dmi(sys);
			this->set_aniso(sys);
			this->set_spc(sys);
			this->set_bqe(sys);
			this->set_fourspin(sys);
			this->set_temper(sys);
			this->set_mu_spin(sys);
			this->set_periodical(this->s);
		}
	}
}

void SettingsWidget::set_hamiltonian_aniso()
{
	if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
	{
		/*this->set_extB_Anisotropic(this->s);
		this->set_aniso(this->s);
		this->set_spc(this->s);
		this->set_temper(this->s);
		this->set_mu_spin(this->s);*/
		this->set_periodical(this->s);
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
	{
		for (auto sys : this->c->images)
		{
			this->set_extB_Anisotropic(sys);
			this->set_spc(sys);
			this->set_temper(sys);
			this->set_mu_spin(sys);
		}
	}
	else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
	{
		for (auto sys : this->c->images)
		{
			this->set_extB_Anisotropic(sys);
			this->set_aniso(sys);
			this->set_spc(sys);
			this->set_temper(sys);
			this->set_mu_spin(sys);
		}
	}
}

void SettingsWidget::set_parameters()
{
	if (this->comboBox_Parameters_ApplyTo->currentText() == "Current Image")
	{
		this->set_dt(this->s);
		this->set_damping(this->s);
		this->set_spring_constant();
		this->set_climbing_falling();
	}
	else if (this->comboBox_Parameters_ApplyTo->currentText() == "Current Image Chain")
	{
		for (auto sys : this->c->images)
		{
			this->set_dt(sys);
			this->set_damping(sys);
			this->set_spring_constant();
			this->set_climbing_falling();
		}
	}
	else if (this->comboBox_Parameters_ApplyTo->currentText() == "All Images")
	{
		for (auto sys : this->c->images)
		{
			this->set_dt(sys);
			this->set_damping(sys);
			this->set_spring_constant();
			this->set_climbing_falling();
		}
	}
}

void SettingsWidget::set_extB(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (this->checkBox_extH->isChecked()) {
			ham->external_field_magnitude = this->lineEdit_extH->text().toDouble()*  ham->mu_s * Utility::Vectormath::MuB();
		}
		else {
			ham->external_field_magnitude = 0.0;
		}
		ham->external_field_normal[0] = lineEdit_extHx->text().toDouble();
		ham->external_field_normal[1] = lineEdit_extHy->text().toDouble();
		ham->external_field_normal[2] = lineEdit_extHz->text().toDouble();
		try {
			Utility::Vectormath::Normalize(ham->external_field_normal);
		}
		catch (int ex) {
			if (ex == 99) {
				ham->external_field_normal[0] = 0.0;
				ham->external_field_normal[1] = 0.0;
				ham->external_field_normal[2] = 1.0;
				Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "B_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_extHx->setText(QString::number(0.0));
				lineEdit_extHy->setText(QString::number(0.0));
				lineEdit_extHz->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}

void SettingsWidget::set_extB_Anisotropic(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();

		if (this->checkBox_extH_aniso->isChecked()) {
			for (int iatom = 0; iatom < s->nos; ++iatom) {
				ham->external_field_magnitude[iatom] = this->lineEdit_extH_aniso->text().toDouble()*  ham->mu_s[iatom] * Utility::Vectormath::MuB();
			}
		}
		else {
			for (int iatom = 0; iatom < s->nos; ++iatom) {
				ham->external_field_magnitude[iatom] = 0.0;
			}
		}
		for (int iatom = 0; iatom < s->nos; ++iatom) {
			ham->external_field_normal[0][iatom] = lineEdit_extHx_aniso->text().toDouble();
			ham->external_field_normal[1][iatom] = lineEdit_extHy_aniso->text().toDouble();
			ham->external_field_normal[2][iatom] = lineEdit_extHz_aniso->text().toDouble();
		}
		try {
			for (int iatom = 0; iatom < s->nos; ++iatom) {
				Utility::Vectormath::Normalize(ham->external_field_normal[iatom]);
			}
		}
		catch (int ex) {
			if (ex == 99) {
				Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "B_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_extHx_aniso->setText(QString::number(0.0));
				lineEdit_extHy_aniso->setText(QString::number(0.0));
				lineEdit_extHz_aniso->setText(QString::number(1.0));
				set_extB_Anisotropic(s);
			}
			else { throw(ex); }
		}
	}
}

void SettingsWidget::set_dmi(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (this->checkBox_dmi->isChecked()) {
			ham->dij = this->lineEdit_dmi->text().toDouble();
		}
		else {
			ham->dij = 0.0;
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}

void SettingsWidget::set_aniso(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (this->checkBox_aniso->isChecked()) {
			ham->anisotropy_magnitude = this->lineEdit_aniso->text().toDouble();
		}
		else {
			ham->anisotropy_magnitude = 0.0;
		}
		ham->anisotropy_normal[0] = lineEdit_anisox->text().toDouble();
		ham->anisotropy_normal[1] = lineEdit_anisoy->text().toDouble();
		ham->anisotropy_normal[2] = lineEdit_anisoz->text().toDouble();
		try {
			Utility::Vectormath::Normalize(ham->anisotropy_normal);
		}
		catch (Utility::Exception ex) {
			if (ex == Utility::Exception::Division_by_zero) {
				ham->anisotropy_normal[0] = 0.0;
				ham->anisotropy_normal[1] = 0.0;
				ham->anisotropy_normal[2] = 1.0;
				Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "Aniso_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_anisox->setText(QString::number(0.0));
				lineEdit_anisoy->setText(QString::number(0.0));
				lineEdit_anisoz->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}
void SettingsWidget::set_spc(std::shared_ptr<Data::Spin_System> s)
{
	if (this->checkBox_spin_torque->isChecked()) {
		s->llg_parameters->stt_magnitude = this->lineEdit_spin_torque->text().toDouble();
	}
	else {
		this->s->llg_parameters->stt_magnitude = 0.0;
	}
	s->llg_parameters->stt_polarisation_normal[0] = lineEdit_spin_torquex->text().toDouble();
	s->llg_parameters->stt_polarisation_normal[1] = lineEdit_spin_torquey->text().toDouble();
	s->llg_parameters->stt_polarisation_normal[2] = lineEdit_spin_torquez->text().toDouble();
	try {
		Utility::Vectormath::Normalize(s->llg_parameters->stt_polarisation_normal);
	}
	catch (Utility::Exception ex) {
		if (ex == Utility::Exception::Division_by_zero) {
			s->llg_parameters->stt_polarisation_normal[0] = 0.0;
			s->llg_parameters->stt_polarisation_normal[1] = 0.0;
			s->llg_parameters->stt_polarisation_normal[2] = 1.0;
			Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "s_c_vec = {0,0,0} replaced by {0,0,1}");
			lineEdit_spin_torquex->setText(QString::number(0.0));
			lineEdit_spin_torquey->setText(QString::number(0.0));
			lineEdit_spin_torquez->setText(QString::number(1.0));
		}
		else { throw(ex); }
	}
}
void SettingsWidget::set_bqe(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (this->checkBox_bqe->isChecked()) {
			ham->bij = this->lineEdit_bqe->text().toDouble();
		}
		else {
			ham->bij = 0.0;
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}
void SettingsWidget::set_fourspin(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (this->checkBox_fourspin->isChecked()) {
			ham->kijkl = this->lineEdit_fourspin->text().toDouble();
		}
		else {
			ham->kijkl = 0.0;
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}
void SettingsWidget::set_temper(std::shared_ptr<Data::Spin_System> s)
{
	s->llg_parameters->temperature = this->lineEdit_temper->text().toDouble();
}

void SettingsWidget::set_periodical(std::shared_ptr<Data::Spin_System> ss)
{
	if (s->is_isotropic)
	{
		ss->hamiltonian->boundary_conditions[0] = this->checkBox_iso_periodical_a->isChecked();
		ss->hamiltonian->boundary_conditions[1] = this->checkBox_iso_periodical_b->isChecked();
		ss->hamiltonian->boundary_conditions[2] = this->checkBox_iso_periodical_c->isChecked();
	}
	else if (!s->is_isotropic)
	{
		ss->hamiltonian->boundary_conditions[0] = this->checkBox_aniso_periodical_a->isChecked();
		ss->hamiltonian->boundary_conditions[1] = this->checkBox_aniso_periodical_b->isChecked();
		ss->hamiltonian->boundary_conditions[2] = this->checkBox_aniso_periodical_c->isChecked();
	}
}

// Reads the Exchange jij values of s into the GUI depending on the number of shells,
// enables/hides the appropriate fields and connects the slots
void SettingsWidget::ReadExchange()
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (ham->n_neigh_shells > 0) {
			lineEdit_exchange1->setText(QString::number(ham->jij[0]));
			lineEdit_exchange1->setEnabled(true);
		}
		else { lineEdit_exchange1->hide(); }
		if (ham->n_neigh_shells > 1) {
			lineEdit_exchange2->setText(QString::number(ham->jij[1]));
			lineEdit_exchange2->setEnabled(true);
		}
		else { lineEdit_exchange2->hide(); }
		if (ham->n_neigh_shells > 2) {
			lineEdit_exchange3->setText(QString::number(ham->jij[2]));
			lineEdit_exchange3->setEnabled(true);
		}
		else { lineEdit_exchange3->hide(); }
		if (ham->n_neigh_shells > 3) {
			lineEdit_exchange4->setText(QString::number(ham->jij[3]));
			lineEdit_exchange4->setEnabled(true);
		}
		else { lineEdit_exchange4->hide(); }
		if (ham->n_neigh_shells > 4) {
			lineEdit_exchange5->setText(QString::number(ham->jij[4]));
			lineEdit_exchange5->setEnabled(true);
		}
		else { lineEdit_exchange5->hide(); }
		checkBox_exchange->setChecked(true);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
	int x = 0;
}

void SettingsWidget::set_exchange(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		if (checkBox_exchange->isChecked())
		{
			if (lineEdit_exchange1->isEnabled()) { ham->jij[0] = lineEdit_exchange1->text().toDouble(); }
			if (lineEdit_exchange2->isEnabled()) { ham->jij[1] = lineEdit_exchange2->text().toDouble(); }
			if (lineEdit_exchange3->isEnabled()) { ham->jij[2] = lineEdit_exchange3->text().toDouble(); }
			if (lineEdit_exchange4->isEnabled()) { ham->jij[3] = lineEdit_exchange4->text().toDouble(); }
			if (lineEdit_exchange5->isEnabled()) { ham->jij[4] = lineEdit_exchange5->text().toDouble(); }
		}
		else {
			for (int i = 0; i < ham->n_neigh_shells; ++i) {
				ham->jij[i] = 0.0;
			}
		}
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}

void SettingsWidget::set_dt(std::shared_ptr<Data::Spin_System> s)
{
	s->llg_parameters->dt = lineEdit_dt->text().toDouble();
}

void SettingsWidget::set_damping(std::shared_ptr<Data::Spin_System> s)
{
	s->llg_parameters->damping = lineEdit_Damping->text().toDouble();
}

void SettingsWidget::set_spring_constant()
{
	c->gneb_parameters->spring_constant = lineEdit_gneb_springconstant->text().toDouble();
}

void SettingsWidget::set_climbing_falling()
{
	c->climbing_image[c->active_image] = radioButton_ClimbingImage->isChecked();
	c->falling_image[c->active_image] = radioButton_FallingImage->isChecked();
}

void SettingsWidget::set_mu_spin(std::shared_ptr<Data::Spin_System> s)
{
	if (s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
		ham->mu_s = lineEdit_muSpin->text().toDouble();
		set_extB(s);
	}
	else if (!s->is_isotropic)
	{
		auto ham = (Engine::Hamiltonian_Anisotropic*)s->hamiltonian.get();
	}
}

void SettingsWidget::print_Energies_to_console()
{
	s->UpdateEnergy();
	std::cout << "E_tot = " << s->E / s->nos << "  ||| Zeeman = ";
	std::cout << s->E_array[ENERGY_POS_ZEEMAN] / s->nos << "  | Aniso = "
		<< s->E_array[ENERGY_POS_ANISOTROPY] / s->nos << "  | Exchange = "
		<< s->E_array[ENERGY_POS_EXCHANGE] / s->nos << "  | DMI = "
		<< s->E_array[ENERGY_POS_DMI] / s->nos << "  | BQC = "
		<< s->E_array[ENERGY_POS_BQC] / s->nos << "  | FourSC = "
		<< s->E_array[ENERGY_POS_FSC] / s->nos << "  | DD = "
		<< s->E_array[ENERGY_POS_DD] / s->nos << std::endl;
}


