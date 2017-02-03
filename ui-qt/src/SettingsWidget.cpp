// #include <QtWidgets>

#include "MainWindow.hpp"
#include "SettingsWidget.hpp"
#include "SpinWidget.hpp"

#include "Interface_Configurations.h"
#include "Interface_Transitions.h"
#include "Interface_Log.h"
#include "Interface_System.h"
#include "Interface_Geometry.h"
#include "Interface_Chain.h"
#include "Interface_Collection.h"
#include "Interface_Hamiltonian.h"
#include "Interface_Parameters.h"
#include "Interface_Exception.h"

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
	if (H_name == "Isotropic Heisenberg") this->tabWidget_Settings->removeTab(3);
	else if (H_name == "Anisotropic Heisenberg") this->tabWidget_Settings->removeTab(2);
	else
	{
		this->tabWidget_Settings->removeTab(2);
		this->tabWidget_Settings->removeTab(2);
	}

	// Load information from Spin Systems
	this->updateData();
	//this->set_visualization_mode();

	// Connect slots
	this->Setup_Configurations_Slots();
	this->Setup_Transitions_Slots();
	this->Setup_Hamiltonian_Isotropic_Slots();
	this->Setup_Hamiltonian_Anisotropic_Slots();
	this->Setup_Parameters_Slots();
	this->Setup_Visualization_Slots();
}

void SettingsWidget::updateData()
{
	// Load Hamiltonian Contents
	std::string H_name = Hamiltonian_Get_Name(state.get());
	if (H_name == "Isotropic Heisenberg") this->Load_Hamiltonian_Isotropic_Contents();
	else if (H_name == "Anisotropic Heisenberg") this->Load_Hamiltonian_Anisotropic_Contents();
	// Load Parameters Contents
	this->Load_Parameters_Contents();
	// ToDo: Also update Debug etc!
	this->Load_Visualization_Contents();
}


// -----------------------------------------------------------------------------------
// -------------- Helpers for fetching Configurations Settings -----------------------
// -----------------------------------------------------------------------------------
std::array<float,3> SettingsWidget::get_position()
{
	return std::array<float,3>
	{
		lineEdit_pos_x->text().toFloat(),
		lineEdit_pos_y->text().toFloat(),
		lineEdit_pos_z->text().toFloat()
	};
}
std::array<float,3> SettingsWidget::get_border_rectangular()
{
	std::array<float,3> ret{-1,-1,-1};
	if (checkBox_border_rectangular_x->isChecked())
		ret[0] = lineEdit_border_x->text().toFloat();
	if (checkBox_border_rectangular_y->isChecked())
		ret[1] = lineEdit_border_y->text().toFloat();
	if (checkBox_border_rectangular_z->isChecked())
		ret[2] = lineEdit_border_z->text().toFloat();
	return ret;
}
float SettingsWidget::get_border_cylindrical()
{
	if (checkBox_border_cylindrical->isChecked())
	{
		return lineEdit_border_cylindrical->text().toFloat();
	}
	else
	{
		return -1;
	}
}
float SettingsWidget::get_border_spherical()
{
	if (checkBox_border_spherical->isChecked())
	{
		return lineEdit_border_spherical->text().toFloat();
	}
	else
	{
		return -1;
	}
}
float SettingsWidget::get_inverted()
{
	return checkBox_inverted->isChecked();
}


// -----------------------------------------------------------------------------------
// --------------------- Configurations and Transitions ------------------------------
// -----------------------------------------------------------------------------------
void SettingsWidget::randomPressed()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Random");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	Configuration_Random(this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}
void SettingsWidget::addNoisePressed()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Add Noise");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}
void SettingsWidget::minusZ()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Minus Z");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	Configuration_MinusZ(this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}
void SettingsWidget::plusZ()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Plus Z");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	Configuration_PlusZ(this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}

void SettingsWidget::create_Hopfion()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Create Hopfion");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	float r = lineEdit_hopfion_radius->text().toFloat();
	int order = lineEdit_hopfion_order->text().toInt();
	Configuration_Hopfion(this->state.get(), r, order, pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}

void SettingsWidget::create_Skyrmion()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Create Skyrmion");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	float rad = lineEdit_skyrmion_radius->text().toFloat();
	float speed = lineEdit_skyrmion_order->text().toFloat();
	float phase = lineEdit_skyrmion_phase->text().toFloat();
	bool upDown = checkBox_skyrmion_UpDown->isChecked();
	bool achiral = checkBox_skyrmion_achiral->isChecked();
	bool rl = checkBox_skyrmion_RL->isChecked();
	// bool experimental = checkBox_sky_experimental->isChecked();
	Configuration_Skyrmion(this->state.get(), rad, speed, phase, upDown, achiral, rl, pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}

void SettingsWidget::create_SpinSpiral()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button createSpinSpiral");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	float direction[3] = { lineEdit_SS_dir_x->text().toFloat(), lineEdit_SS_dir_y->text().toFloat(), lineEdit_SS_dir_z->text().toFloat() };
	float axis[3] = { lineEdit_SS_axis_x->text().toFloat(), lineEdit_SS_axis_y->text().toFloat(), lineEdit_SS_axis_z->text().toFloat() };
	float period = lineEdit_SS_period->text().toFloat();
	const char * direction_type;
	if (comboBox_SS->currentText() == "Real Lattice") direction_type = "Real Lattice";
	else if (comboBox_SS->currentText() == "Reciprocal Lattice") direction_type = "Reciprocal Lattice";
	else if (comboBox_SS->currentText() == "Real Space") direction_type = "Real Space";
	Configuration_SpinSpiral(this->state.get(), direction_type, direction, axis, period, pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}

void SettingsWidget::domainPressed()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "button Domain");
	// Get settings
	auto pos = get_position();
	auto border_rect = get_border_rectangular();
	float border_cyl = get_border_cylindrical();
	float border_sph = get_border_spherical();
	bool inverted = get_inverted();
	// Create configuration
	float dir[3] = { lineEdit_domain_dir_x->text().toFloat(), lineEdit_domain_dir_y->text().toFloat(), lineEdit_domain_dir_z->text().toFloat() };
	Configuration_Domain(this->state.get(), dir, pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

	// Optionally add noise
	this->configurationAddNoise();
	print_Energies_to_console();
	Chain_Update_Data(this->state.get());
	this->_spinWidget->updateData();
}

void SettingsWidget::configurationAddNoise()
{
	// Add Noise
	if (this->checkBox_Configuration_Noise->isChecked())
	{
		// Get settings
		auto pos = get_position();
		auto border_rect = get_border_rectangular();
		float border_cyl = get_border_cylindrical();
		float border_sph = get_border_spherical();
		bool inverted = get_inverted();
		// Create configuration
		float temperature = lineEdit_Configuration_Noise->text().toFloat();
		Configuration_Add_Noise_Temperature(this->state.get(), temperature, pos.data(), border_rect.data(), border_cyl, border_sph, inverted);

		Chain_Update_Data(this->state.get());
		this->_spinWidget->updateData();
	}
}

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


void SettingsWidget::Load_Parameters_Contents()
{
	float d;
	int image_type;
	int i;

	// LLG Damping
	Parameters_Get_LLG_Damping(state.get(), &d);
	this->lineEdit_Damping->setText(QString::number(d));
	// Converto to PicoSeconds
	Parameters_Get_LLG_Time_Step(state.get(), &d);
	this->lineEdit_dt->setText(QString::number(d));
	// LLG Iteration Params
	i = Parameters_Get_LLG_N_Iterations(state.get());
	this->lineEdit_llg_n_iterations->setText(QString::number(i));
	i = Parameters_Get_LLG_N_Iterations_Log(state.get());
	this->lineEdit_llg_log_steps->setText(QString::number(i));
	// GNEB Interation Params
	i = Parameters_Get_GNEB_N_Iterations(state.get());
	this->lineEdit_gneb_n_iterations->setText(QString::number(i));
	i = Parameters_Get_GNEB_N_Iterations_Log(state.get());
	this->lineEdit_gneb_log_steps->setText(QString::number(i));

	// GNEB Spring Constant
	Parameters_Get_GNEB_Spring_Constant(state.get(), &d);
	this->lineEdit_gneb_springconstant->setText(QString::number(d));

	// Normal/Climbing/Falling image radioButtons
	Parameters_Get_GNEB_Climbing_Falling(state.get(), &image_type);
	if (image_type == 0)
		this->radioButton_Normal->setChecked(true);
	else if (image_type == 1)
		this->radioButton_ClimbingImage->setChecked(true);
	else if (image_type == 2)
		this->radioButton_FallingImage->setChecked(true);
	else if (image_type == 3)
		this->radioButton_Stationary->setChecked(true);
}


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

	// Spin polarized current (does not really belong to interactions)
	Hamiltonian_Get_STT(state.get(), &d, vd);
	this->lineEdit_spin_torque->setText(QString::number(d));
	this->lineEdit_spin_torquex->setText(QString::number(vd[0]));
	this->lineEdit_spin_torquey->setText(QString::number(vd[1]));
	this->lineEdit_spin_torquez->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_spin_torque->setChecked(true);

	// BQE
	Hamiltonian_Get_BQE(state.get(), &d);
	this->lineEdit_bqe->setText(QString::number(d));
	if (d > 0.0) this->checkBox_bqe->setChecked(true);

	// FourSpin
	Hamiltonian_Get_FSC(state.get(), &d);
	this->lineEdit_fourspin->setText(QString::number(d));
	if (d > 0.0) this->checkBox_fourspin->setChecked(true);

	// Temperature (does not really belong to interactions)
	Hamiltonian_Get_Temperature(state.get(), &d);
	this->lineEdit_temper->setText(QString::number(d));
	if (d > 0.0) this->checkBox_Temperature->setChecked(true);
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

	// Spin polarised current
	Hamiltonian_Get_STT(state.get(), &d, vd);
	this->lineEdit_stt_aniso->setText(QString::number(d));
	this->lineEdit_sttx_aniso->setText(QString::number(vd[0]));
	this->lineEdit_stty_aniso->setText(QString::number(vd[1]));
	this->lineEdit_sttz_aniso->setText(QString::number(vd[2]));
	if (d > 0.0) this->checkBox_stt_aniso->setChecked(true);

	// Temperature
	Hamiltonian_Get_Temperature(state.get(), &d);
	this->lineEdit_T_aniso->setText(QString::number(d));
	if (d > 0.0) this->checkBox_T_aniso->setChecked(true);
}

void SettingsWidget::Load_Visualization_Contents()
{
	// Mode
	if (this->_spinWidget->visualizationMode() == SpinWidget::VisualizationMode::SYSTEM)
		this->radioButton_vismode_system->setChecked(true);
	else
		this->radioButton_vismode_sphere->setChecked(true);
	
	// System
	this->checkBox_show_arrows->setChecked(_spinWidget->show_arrows);
	this->checkBox_showBoundingBox->setChecked(_spinWidget->show_boundingbox);
	this->checkBox_show_surface->setChecked(_spinWidget->show_surface);
	this->checkBox_show_isosurface->setChecked(_spinWidget->show_isosurface);
	this->checkBox_isosurfaceshadows->setChecked(_spinWidget->isosurfaceshadows());

	// Miniview
	this->checkBox_showMiniView->setChecked(_spinWidget->isMiniviewEnabled());
	this->comboBox_miniViewPosition->setCurrentIndex((int)_spinWidget->miniviewPosition());
	
	// Coordinate System
	this->checkBox_showCoordinateSystem->setChecked(_spinWidget->isCoordinateSystemEnabled());
	this->comboBox_coordinateSystemPosition->setCurrentIndex((int)_spinWidget->coordinateSystemPosition());

	// Z Range Arrows
	auto z_range = _spinWidget->zRange();
	if (z_range.x < -1)
		z_range.x = -1;
	if (z_range.x > 1)
		z_range.x = 1;
	if (z_range.y < -1)
		z_range.y = -1;
	if (z_range.y > 1)
		z_range.y = 1;
	
	// Overall filter
	horizontalSlider_overall_zmin->setInvertedAppearance(true);
	horizontalSlider_overall_zmin->setRange(-100, 100);
	horizontalSlider_overall_zmin->setValue((int)(-z_range.x * 100));
	horizontalSlider_overall_zmax->setRange(-100, 100);
	horizontalSlider_overall_zmax->setValue((int)(z_range.y * 100));
	horizontalSlider_overall_zmin->setTracking(true);
	horizontalSlider_overall_zmax->setTracking(true);


	// X Range Surface
	horizontalSlider_surface_xmin->setRange(1, 99999);
	horizontalSlider_surface_xmin->setValue((int)(1));
	horizontalSlider_surface_xmax->setRange(1, 99999);
	horizontalSlider_surface_xmax->setValue((int)(99999));
	horizontalSlider_surface_xmin->setTracking(true);
	horizontalSlider_surface_xmax->setTracking(true);
	// Y Range Surface
	horizontalSlider_surface_ymin->setRange(1, 99999);
	horizontalSlider_surface_ymin->setValue((int)(1));
	horizontalSlider_surface_ymax->setRange(1, 99999);
	horizontalSlider_surface_ymax->setValue((int)(99999));
	horizontalSlider_surface_ymin->setTracking(true);
	horizontalSlider_surface_ymax->setTracking(true);
	// Z Range Surface
	horizontalSlider_surface_zmin->setRange(1, 99999);
	horizontalSlider_surface_zmin->setValue((int)(1));
	horizontalSlider_surface_zmax->setRange(1, 99999);
	horizontalSlider_surface_zmax->setValue((int)(99999));
	horizontalSlider_surface_zmin->setTracking(true);
	horizontalSlider_surface_zmax->setTracking(true);
  
	// Isosurface
	auto isovalue = _spinWidget->isovalue();
	horizontalSlider_isovalue->setRange(0, 100);
	horizontalSlider_isovalue->setValue((int)(isovalue+1*50));
	int component = _spinWidget->isocomponent();
	if (component == 0) this->radioButton_isosurface_x->setChecked(true);
	else if (component == 1) this->radioButton_isosurface_y->setChecked(true);
	else if (component == 2) this->radioButton_isosurface_z->setChecked(true);

	// Colormap
	int idx_cm = (int)_spinWidget->colormap();
	comboBox_colormap->setCurrentIndex(idx_cm);

	// Perspective / FOV
	if (_spinWidget->verticalFieldOfView() == 0)
	{
		radioButton_orthographicProjection->setChecked(true);
	}
	else
	{
		radioButton_orthographicProjection->setChecked(false);
	}


	// Arrows: size and lod
	horizontalSlider_arrowsize->setRange(0, 20);
	float logs = std::log10(_spinWidget->arrowSize());
	horizontalSlider_arrowsize->setValue((int)((logs+1)*10));
	lineEdit_arrows_lod->setText(QString::number(_spinWidget->arrowLOD()));

	// Sphere
	horizontalSlider_spherePointSize->setRange(1, 10);
	horizontalSlider_spherePointSize->setValue((int)_spinWidget->spherePointSizeRange().y);

	// Bounding Box
	//checkBox_showBoundingBox->setChecked(_spinWidget->isBoundingBoxEnabled());

	// Background
	int idx_bg = (int)_spinWidget->backgroundColor();
	comboBox_backgroundColor->setCurrentIndex(idx_bg);

	// Camera
	this->read_camera();
}

// -----------------------------------------------------------------------------------
// --------------------- Setters for Hamiltonians and Parameters ---------------------
// -----------------------------------------------------------------------------------


void SettingsWidget::set_parameters()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d;
		int i;

		// Time step [ps]
		// dt = time_step [ps] * 10^-12 * gyromagnetic raio / mu_B  { / (1+damping^2)} <- not implemented
		d = this->lineEdit_dt->text().toFloat();
		Parameters_Set_LLG_Time_Step(state.get(), d, idx_image, idx_chain);
		
		// Damping
		d = this->lineEdit_Damping->text().toFloat();
		Parameters_Set_LLG_Damping(state.get(), d);
		// n iterations
		i = this->lineEdit_llg_n_iterations->text().toInt();
		Parameters_Set_LLG_N_Iterations(state.get(), i);
		i = this->lineEdit_gneb_n_iterations->text().toInt();
		Parameters_Set_GNEB_N_Iterations(state.get(), i);
		// log steps
		i = this->lineEdit_llg_log_steps->text().toInt();
		Parameters_Set_LLG_N_Iterations_Log(state.get(), i);
		i = this->lineEdit_gneb_log_steps->text().toInt();
		Parameters_Set_GNEB_N_Iterations_Log(state.get(), i);
		// Spring Constant
		d = this->lineEdit_gneb_springconstant->text().toFloat();
		Parameters_Set_GNEB_Spring_Constant(state.get(), d);
		// Climbing/Falling Image
		int image_type = 0;
		if (this->radioButton_ClimbingImage->isChecked())
			image_type = 1;
		if (this->radioButton_FallingImage->isChecked())
			image_type = 2;
		if (this->radioButton_Stationary->isChecked())
			image_type = 3;
		Parameters_Set_GNEB_Climbing_Falling(state.get(), image_type, idx_image, idx_chain);
	};

	if (this->comboBox_Parameters_ApplyTo->currentText() == "Current Image")
	{
		apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
	}
	else if (this->comboBox_Parameters_ApplyTo->currentText() == "Current Image Chain")
	{
		for (int img=0; img<Chain_Get_NOI(state.get()); ++img)
		{
			apply(img, Chain_Get_Index(state.get()));
		}
	}
	else if (this->comboBox_Parameters_ApplyTo->currentText() == "All Images")
	{
		for (int ich=0; ich<Collection_Get_NOC(state.get()); ++ich)
		{
			for (int img=0; img<Chain_Get_NOI(state.get(),ich); ++img)
			{
				apply(img, ich);
			}
		}
	}
}


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

		// These belong in Parameters, not Hamiltonian
		// Spin polarised current
		if (this->checkBox_spin_torque->isChecked()) {
			d = this->lineEdit_spin_torque->text().toFloat();
		}
		else {
			d = 0.0;
		}
		vd[0] = lineEdit_spin_torquex->text().toFloat();
		vd[1] = lineEdit_spin_torquey->text().toFloat();
		vd[2] = lineEdit_spin_torquez->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "s_c_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_spin_torquex->setText(QString::number(0.0));
				lineEdit_spin_torquey->setText(QString::number(0.0));
				lineEdit_spin_torquez->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_STT(state.get(), d, vd, idx_image, idx_chain);

		// Temperature
		if (this->checkBox_Temperature->isChecked())
			d = this->lineEdit_temper->text().toFloat();
		else
			d = 0.0;
		Hamiltonian_Set_Temperature(state.get(), d, idx_image, idx_chain);
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

void SettingsWidget::set_hamiltonian_aniso_stt()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3];

		// TODO: Make these anisotropic for Anisotropic Hamiltonian
		//		 or move them to Parameters...
		// Spin polarised current
		if (this->checkBox_stt_aniso->isChecked())
			d = this->lineEdit_stt_aniso->text().toFloat();
		else d = 0.0;
		vd[0] = lineEdit_sttx_aniso->text().toFloat();
		vd[1] = lineEdit_stty_aniso->text().toFloat();
		vd[2] = lineEdit_sttz_aniso->text().toFloat();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "s_c_vec = {0,0,0} replaced by {0,0,1}");
				lineEdit_sttx_aniso->setText(QString::number(0.0));
				lineEdit_stty_aniso->setText(QString::number(0.0));
				lineEdit_sttz_aniso->setText(QString::number(1.0));
			}
			else { throw(ex); }
		}
		Hamiltonian_Set_STT(state.get(), d, vd, idx_image, idx_chain);
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

void SettingsWidget::set_hamiltonian_aniso_temp()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d = 0.0;

		// Temperature
		if (this->checkBox_T_aniso->isChecked())
			d = this->lineEdit_T_aniso->text().toFloat();
		Hamiltonian_Set_Temperature(state.get(), d, idx_image, idx_chain);
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
// --------------------- Visualization -----------------------------------------------
// -----------------------------------------------------------------------------------
void SettingsWidget::set_visualisation_source()
{
	this->_spinWidget->setVisualisationSource(this->comboBox_VisualisationSource->currentIndex());
}

void SettingsWidget::set_visualization_mode()
{
	SpinWidget::VisualizationMode mode;

	if (this->radioButton_vismode_sphere->isChecked())
		mode = SpinWidget::VisualizationMode::SPHERE;
	else
		mode = SpinWidget::VisualizationMode::SYSTEM;
	
	this->_spinWidget->setVisualizationMode(mode);
}

void SettingsWidget::set_visualization_perspective()
{
	// Perspective / FOV
	if (radioButton_orthographicProjection->isChecked())
	{
		_spinWidget->setVerticalFieldOfView(0);
	}
	else
	{
		_spinWidget->setVerticalFieldOfView(45);
	}
}

void SettingsWidget::set_visualization_miniview()
{
	bool miniview;
	SpinWidget::WidgetLocation pos;

	miniview = this->checkBox_showMiniView->isChecked();
	if (this->comboBox_miniViewPosition->currentText() == "Bottom Left")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Bottom Right")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Top Left")
	{
		pos = SpinWidget::WidgetLocation::TOP_LEFT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Top Right")
	{
		pos = SpinWidget::WidgetLocation::TOP_RIGHT;
	}

	this->_spinWidget->setVisualizationMiniview(miniview, pos);
}

void SettingsWidget::set_visualization_coordinatesystem()
{
	bool coordinatesystem;
	SpinWidget::WidgetLocation pos;

	coordinatesystem = this->checkBox_showCoordinateSystem->isChecked();
	if (this->comboBox_coordinateSystemPosition->currentText() == "Bottom Left")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Bottom Right")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Top Left")
	{
		pos = SpinWidget::WidgetLocation::TOP_LEFT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Top Right")
	{
		pos = SpinWidget::WidgetLocation::TOP_RIGHT;
	}

	this->_spinWidget->setVisualizationCoordinatesystem(coordinatesystem, pos);
}

void SettingsWidget::set_visualization_system()
{
	bool arrows, boundingbox, surface, isosurface;

	arrows = this->checkBox_show_arrows->isChecked();
	boundingbox = this->checkBox_showBoundingBox->isChecked();
	surface = this->checkBox_show_surface->isChecked();
	isosurface = this->checkBox_show_isosurface->isChecked();

	this->_spinWidget->enableSystem(arrows, boundingbox, surface, isosurface);
}

void SettingsWidget::set_visualization_system_arrows()
{
	float exponent = horizontalSlider_arrowsize->value() / 10.0f - 1.0f;
	float arrowsize = std::pow(10.0f, exponent);
	int arrowlod = lineEdit_arrows_lod->text().toInt();
	this->_spinWidget->setArrows(arrowsize, arrowlod);
}
void SettingsWidget::set_visualization_system_boundingbox()
{

}
void SettingsWidget::set_visualization_system_surface()
{
	float bounds_min[3], bounds_max[3];
	Geometry_Get_Bounds(state.get(), bounds_min, bounds_max);
	float s_min, s_max;

	// X
	s_min = horizontalSlider_surface_xmin->value();
	s_max = horizontalSlider_surface_xmax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_xmin->blockSignals(true);
	horizontalSlider_surface_xmax->blockSignals(true);
	horizontalSlider_surface_xmin->setValue((int)(s_min));
	horizontalSlider_surface_xmax->setValue((int)(s_max));
	horizontalSlider_surface_xmin->blockSignals(false);
	horizontalSlider_surface_xmax->blockSignals(false);
	float x_min = bounds_min[0] + (s_min / 100000.0) * (bounds_max[0] - bounds_min[0]);
	float x_max = bounds_min[0] + (s_max / 100000.0) * (bounds_max[0] - bounds_min[0]);
	// Y
	s_min = horizontalSlider_surface_ymin->value();
	s_max = horizontalSlider_surface_ymax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_ymin->blockSignals(true);
	horizontalSlider_surface_ymax->blockSignals(true);
	horizontalSlider_surface_ymin->setValue((int)(s_min));
	horizontalSlider_surface_ymax->setValue((int)(s_max));
	horizontalSlider_surface_ymin->blockSignals(false);
	horizontalSlider_surface_ymax->blockSignals(false);
	float y_min = bounds_min[1] + (s_min / 100000.0) * (bounds_max[1] - bounds_min[1]);
	float y_max = bounds_min[1] + (s_max / 100000.0) * (bounds_max[1] - bounds_min[1]);
	// Z
	s_min = horizontalSlider_surface_zmin->value();
	s_max = horizontalSlider_surface_zmax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_zmin->blockSignals(true);
	horizontalSlider_surface_zmax->blockSignals(true);
	horizontalSlider_surface_zmin->setValue((int)(s_min));
	horizontalSlider_surface_zmax->setValue((int)(s_max));
	horizontalSlider_surface_zmin->blockSignals(false);
	horizontalSlider_surface_zmax->blockSignals(false);
	float z_min = bounds_min[2] + (s_min / 100000.0) * (bounds_max[2] - bounds_min[2]);
	float z_max = bounds_min[2] + (s_max / 100000.0) * (bounds_max[2] - bounds_min[2]);

	glm::vec2 x_range(x_min, x_max);
	glm::vec2 y_range(y_min, y_max);
	glm::vec2 z_range(z_min, z_max);
	_spinWidget->setSurface(x_range, y_range, z_range);
}

void SettingsWidget::set_visualization_system_overall()
{
	float z_range_min = -horizontalSlider_overall_zmin->value() / 100.0;
	float z_range_max = horizontalSlider_overall_zmax->value() / 100.0;
	if (z_range_min > z_range_max)
	{
		float t = z_range_min;
		z_range_min = z_range_max;
		z_range_max = t;
	}
	horizontalSlider_overall_zmin->blockSignals(true);
	horizontalSlider_overall_zmax->blockSignals(true);
	horizontalSlider_overall_zmin->setValue((int)(-z_range_min * 100));
	horizontalSlider_overall_zmax->setValue((int)(z_range_max * 100));
	horizontalSlider_overall_zmin->blockSignals(false);
	horizontalSlider_overall_zmax->blockSignals(false);

	glm::vec2 z_range(z_range_min, z_range_max);
	_spinWidget->setZRange(z_range);
}

void SettingsWidget::set_visualization_system_isosurface()
{
	_spinWidget->setIsosurfaceshadows(this->checkBox_isosurfaceshadows->isChecked());
}



void SettingsWidget::set_visualization_isovalue_fromslider()
{
	float isovalue = horizontalSlider_isovalue->value() / 50.0f - 1.0f;
	this->lineEdit_isovalue->setText(QString::number(isovalue));
	_spinWidget->setIsovalue(isovalue);
}

void SettingsWidget::set_visualization_isovalue_fromlineedit()
{
	float isovalue = this->lineEdit_isovalue->text().toFloat();
	this->horizontalSlider_isovalue->setValue((int)(isovalue*50 + 50));
	_spinWidget->setIsovalue(isovalue);
}

void SettingsWidget::set_visualization_isocomponent()
{
	if (radioButton_isosurface_x->isChecked())
		_spinWidget->setIsocomponent(0);
	else if (radioButton_isosurface_y->isChecked())
		_spinWidget->setIsocomponent(1);
	else if (radioButton_isosurface_z->isChecked())
		_spinWidget->setIsocomponent(2);
}


void SettingsWidget::set_visualization_sphere()
{
	// This function does not make any sense, does it?
	// Only possibility: draw/dont draw the sphere, only draw the points
}
void SettingsWidget::set_visualization_sphere_pointsize()
{
	this->_spinWidget->setSpherePointSizeRange({ 0.2, this->horizontalSlider_spherePointSize->value() });
}

void SettingsWidget::set_visualization_colormap()
{
	SpinWidget::Colormap colormap = SpinWidget::Colormap::HSV;
	if (comboBox_colormap->currentText() == "HSV, no z-component")
	{
		colormap = SpinWidget::Colormap::HSV_NO_Z;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_RED;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-Green-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_GREEN_RED;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-White-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_WHITE_RED;
	}
	if (comboBox_colormap->currentText() == "White")
	{
		colormap = SpinWidget::Colormap::WHITE;
	}
	if (comboBox_colormap->currentText() == "Gray")
	{
		colormap = SpinWidget::Colormap::GRAY;
	}
	if (comboBox_colormap->currentText() == "Black")
	{
		colormap = SpinWidget::Colormap::BLACK;
	}
	_spinWidget->setColormap(colormap);
}

void SettingsWidget::set_visualization_background()
{
	SpinWidget::Color color;
	SpinWidget::Color invcolor;
	if (comboBox_backgroundColor->currentText() == "Black")
	{
		color = SpinWidget::Color::BLACK;
		invcolor = SpinWidget::Color::WHITE;
	}
	else if (comboBox_backgroundColor->currentText() == "Gray")
	{
		color = SpinWidget::Color::GRAY;
		invcolor = SpinWidget::Color::WHITE;
	}
	else
	{
		color = SpinWidget::Color::WHITE;
		invcolor = SpinWidget::Color::BLACK;
	}
	_spinWidget->setBackgroundColor(color);
	_spinWidget->setBoundingBoxColor(invcolor);
}

// -----------------------------------------------------------------------------------
// --------------------- Camera ------------------------------------------------------
// -----------------------------------------------------------------------------------

void SettingsWidget::set_camera()
{
	set_camera_position();
	set_camera_focus();
	set_camera_upvector();
}

void SettingsWidget::read_camera()
{
    auto camera_position = _spinWidget->getCameraPositon();
	auto center_position = _spinWidget->getCameraFocus();
	auto up_vector = _spinWidget->getCameraUpVector();

	this->lineEdit_camera_pos_x->setText(QString::number(camera_position.x, 'f', 2));
	this->lineEdit_camera_pos_y->setText(QString::number(camera_position.y, 'f', 2));
	this->lineEdit_camera_pos_z->setText(QString::number(camera_position.z, 'f', 2));
	this->lineEdit_camera_focus_x->setText(QString::number(center_position.x, 'f', 2));
	this->lineEdit_camera_focus_y->setText(QString::number(center_position.y, 'f', 2));
	this->lineEdit_camera_focus_z->setText(QString::number(center_position.z, 'f', 2));
	this->lineEdit_camera_upvector_x->setText(QString::number(up_vector.x, 'f', 2));
	this->lineEdit_camera_upvector_y->setText(QString::number(up_vector.y, 'f', 2));
	this->lineEdit_camera_upvector_z->setText(QString::number(up_vector.z, 'f', 2));
}

void SettingsWidget::set_camera_position()
{
	float x = this->lineEdit_camera_pos_x->text().toFloat();
	float y = this->lineEdit_camera_pos_y->text().toFloat();
	float z = this->lineEdit_camera_pos_z->text().toFloat();
    this->_spinWidget->setCameraPositon({x, y, z});
}

void SettingsWidget::set_camera_focus()
{
	float x = this->lineEdit_camera_focus_x->text().toFloat();
	float y = this->lineEdit_camera_focus_y->text().toFloat();
	float z = this->lineEdit_camera_focus_z->text().toFloat();
    this->_spinWidget->setCameraFocus({x, y, z});
}

void SettingsWidget::set_camera_upvector()
{
	float x = this->lineEdit_camera_upvector_x->text().toFloat();
	float y = this->lineEdit_camera_upvector_y->text().toFloat();
	float z = this->lineEdit_camera_upvector_z->text().toFloat();
    this->_spinWidget->setCameraUpVector({x, y, z});
}


// -----------------------------------------------------------------------------------
// --------------------- Utilities ---------------------------------------------------
// -----------------------------------------------------------------------------------


void SettingsWidget::SelectTab(int index)
{
	this->tabWidget_Settings->setCurrentIndex(index);
}


void SettingsWidget::print_Energies_to_console()
{
	System_Update_Data(state.get());
	System_Print_Energy_Array(state.get());
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
	// Spin Torque (does not really belong to interactions)
	connect(this->checkBox_spin_torque, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torque, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquex, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquey, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_spin_torquez, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));
	// Temperature (does not really belong to interactions)
	connect(this->checkBox_Temperature, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_iso()));
	connect(this->lineEdit_temper, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_iso()));

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
	// Spin polarised current
	connect(this->checkBox_stt_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_stt()));
	connect(this->lineEdit_stt_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_stt()));
	connect(this->lineEdit_sttx_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_stt()));
	connect(this->lineEdit_stty_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_stt()));
	connect(this->lineEdit_sttz_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_stt()));
	// Temperature
	connect(this->checkBox_T_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_hamiltonian_aniso_temp()));
	connect(this->lineEdit_T_aniso, SIGNAL(returnPressed()), this, SLOT(set_hamiltonian_aniso_temp()));
}

void SettingsWidget::Setup_Parameters_Slots()
{
	// LLG Damping
	connect(this->lineEdit_Damping, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_dt, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// LLG iteration params
	connect(this->lineEdit_llg_n_iterations, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_llg_log_steps, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// GNEB iteration params
	connect(this->lineEdit_llg_n_iterations, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_llg_log_steps, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// GNEB Spring Constant
	connect(this->lineEdit_gneb_springconstant, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// Normal/Climbing/Falling image radioButtons
	connect(this->radioButton_Normal, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_ClimbingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_FallingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_Stationary, SIGNAL(clicked()), this, SLOT(set_parameters()));
}

void SettingsWidget::Setup_Configurations_Slots()
{
	// Random
	connect(this->pushButton_Random, SIGNAL(clicked()), this, SLOT(randomPressed()));
	// Add Noise
	connect(this->pushButton_AddNoise, SIGNAL(clicked()), this, SLOT(addNoisePressed()));
	// Domain
	connect(this->pushButton_domain, SIGNAL(clicked()), this, SLOT(domainPressed()));
	// Homogeneous
	connect(this->pushButton_plusZ, SIGNAL(clicked()), this, SLOT(plusZ()));
	connect(this->pushButton_minusZ, SIGNAL(clicked()), this, SLOT(minusZ()));
	// Hopfion
	connect(this->pushButton_hopfion, SIGNAL(clicked()), this, SLOT(create_Hopfion()));
	// Skyrmion
	connect(this->pushButton_skyrmion, SIGNAL(clicked()), this, SLOT(create_Skyrmion()));
	// Spin Spiral
	connect(this->pushButton_SS, SIGNAL(clicked()), this, SLOT(create_SpinSpiral()));

	// Domain  LineEdits
	connect(this->lineEdit_domain_dir_x, SIGNAL(returnPressed()), this, SLOT(domainPressed()));
	connect(this->lineEdit_domain_dir_y, SIGNAL(returnPressed()), this, SLOT(domainPressed()));
	connect(this->lineEdit_domain_dir_z, SIGNAL(returnPressed()), this, SLOT(domainPressed()));

	// Hopfion LineEdits
	connect(this->lineEdit_hopfion_radius, SIGNAL(returnPressed()), this, SLOT(create_Hopfion()));
	connect(this->lineEdit_hopfion_order, SIGNAL(returnPressed()), this, SLOT(create_Hopfion()));

	// Skyrmion LineEdits
	connect(this->lineEdit_skyrmion_order, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_skyrmion_phase, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));
	connect(this->lineEdit_skyrmion_radius, SIGNAL(returnPressed()), this, SLOT(create_Skyrmion()));

	// SpinSpiral LineEdits
	connect(this->lineEdit_SS_dir_x, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_dir_y, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_dir_z, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_x, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_y, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_axis_z, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));
	connect(this->lineEdit_SS_period, SIGNAL(returnPressed()), this, SLOT(create_SpinSpiral()));

}

void SettingsWidget::Setup_Transitions_Slots()
{
	// Homogeneous Transition
	connect(this->pushButton_Transition_Homogeneous, SIGNAL(clicked()), this, SLOT(homogeneousTransitionPressed()));
}

void SettingsWidget::Setup_Visualization_Slots()
{
	connect(comboBox_VisualisationSource, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualisation_source()));
	// Mode
	connect(radioButton_vismode_sphere, SIGNAL(toggled(bool)), this, SLOT(set_visualization_mode()));
	connect(radioButton_vismode_system, SIGNAL(toggled(bool)), this, SLOT(set_visualization_mode()));
	connect(radioButton_perspectiveProjection, SIGNAL(toggled(bool)), this, SLOT(set_visualization_perspective()));
	connect(radioButton_orthographicProjection, SIGNAL(toggled(bool)), this, SLOT(set_visualization_perspective()));
	// Miniview
	connect(checkBox_showMiniView, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_miniview()));
	connect(comboBox_miniViewPosition, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_miniview()));
	// Coordinate System
	connect(checkBox_showCoordinateSystem, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_coordinatesystem()));
	connect(comboBox_coordinateSystemPosition, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_coordinatesystem()));
	// System
	connect(checkBox_show_arrows, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_showBoundingBox, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_show_surface, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_show_isosurface, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	//		arrows
	connect(horizontalSlider_arrowsize, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_arrows()));
	connect(lineEdit_arrows_lod, SIGNAL(returnPressed()), this, SLOT(set_visualization_system_arrows()));
	//		bounding box
	//		surface
	connect(horizontalSlider_surface_xmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_xmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_ymin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_ymax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	//		overall
	connect(horizontalSlider_overall_zmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall()));
	connect(horizontalSlider_overall_zmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall()));
	//		isosurface
	connect(horizontalSlider_isovalue, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_isovalue_fromslider()));
	connect(this->lineEdit_isovalue, SIGNAL(returnPressed()), this, SLOT(set_visualization_isovalue_fromlineedit()));
	connect(radioButton_isosurface_x, SIGNAL(toggled(bool)), this, SLOT(set_visualization_isocomponent()));
	connect(radioButton_isosurface_y, SIGNAL(toggled(bool)), this, SLOT(set_visualization_isocomponent()));
	connect(radioButton_isosurface_z, SIGNAL(toggled(bool)), this, SLOT(set_visualization_isocomponent()));
	connect(checkBox_isosurfaceshadows, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system_isosurface()));
	// Sphere
	connect(horizontalSlider_spherePointSize, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_sphere_pointsize()));
	// Colors
	connect(comboBox_backgroundColor, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_background()));
	connect(comboBox_colormap, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_colormap()));
	// Camera
	connect(this->lineEdit_camera_pos_x, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_pos_y, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_pos_z, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_focus_x, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_focus_y, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_focus_z, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_upvector_x, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->lineEdit_camera_upvector_y, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->lineEdit_camera_upvector_z, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->pushButton_set_camera, SIGNAL(clicked()), this, SLOT(set_camera()));
	connect(this->pushButton_read_camera, SIGNAL(clicked()), this, SLOT(read_camera()));
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
	//		spin polarised current
	this->lineEdit_spin_torque->setValidator(this->number_validator);
	this->lineEdit_spin_torquex->setValidator(this->number_validator);
	this->lineEdit_spin_torquey->setValidator(this->number_validator);
	this->lineEdit_spin_torquez->setValidator(this->number_validator);
	//		BQE
	this->lineEdit_bqe->setValidator(this->number_validator);
	//		FSC
	this->lineEdit_fourspin->setValidator(this->number_validator);
	//		temperature
	this->lineEdit_temper->setValidator(this->number_validator_unsigned);

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
	//		spin polarised current
	this->lineEdit_stt_aniso->setValidator(this->number_validator);
	this->lineEdit_sttx_aniso->setValidator(this->number_validator);
	this->lineEdit_stty_aniso->setValidator(this->number_validator);
	this->lineEdit_sttz_aniso->setValidator(this->number_validator);
	//		temperature
	this->lineEdit_T_aniso->setValidator(this->number_validator_unsigned);

	// Configurations
	//		Settings
	this->lineEdit_Configuration_Noise->setValidator(this->number_validator_unsigned);
	this->lineEdit_pos_x->setValidator(this->number_validator);
	this->lineEdit_pos_y->setValidator(this->number_validator);
	this->lineEdit_pos_z->setValidator(this->number_validator);
	this->lineEdit_border_x->setValidator(this->number_validator_unsigned);
	this->lineEdit_border_y->setValidator(this->number_validator_unsigned);
	this->lineEdit_border_z->setValidator(this->number_validator_unsigned);
	this->lineEdit_border_cylindrical->setValidator(this->number_validator_unsigned);
	this->lineEdit_border_spherical->setValidator(this->number_validator_unsigned);
	//		Hopfion
	this->lineEdit_hopfion_radius->setValidator(this->number_validator);
	this->lineEdit_hopfion_order->setValidator(this->number_validator_int_unsigned);
	//		Skyrmion
	this->lineEdit_skyrmion_order->setValidator(this->number_validator_int_unsigned);
	this->lineEdit_skyrmion_phase->setValidator(this->number_validator);
	this->lineEdit_skyrmion_radius->setValidator(this->number_validator);
	//		Spin Spiral
	this->lineEdit_SS_dir_x->setValidator(this->number_validator);
	this->lineEdit_SS_dir_y->setValidator(this->number_validator);
	this->lineEdit_SS_dir_z->setValidator(this->number_validator);
	this->lineEdit_SS_axis_x->setValidator(this->number_validator);
	this->lineEdit_SS_axis_y->setValidator(this->number_validator);
	this->lineEdit_SS_axis_z->setValidator(this->number_validator);
	this->lineEdit_SS_period->setValidator(this->number_validator);
	//		Domain
	this->lineEdit_domain_dir_x->setValidator(this->number_validator);
	this->lineEdit_domain_dir_y->setValidator(this->number_validator);
	this->lineEdit_domain_dir_z->setValidator(this->number_validator);

	// Transitions
	this->lineEdit_Transition_Noise->setValidator(this->number_validator_unsigned);
	this->lineEdit_Transition_Homogeneous_First->setValidator(this->number_validator_int_unsigned);
	this->lineEdit_Transition_Homogeneous_Last->setValidator(this->number_validator_int_unsigned);

	// Parameters
	//		LLG
	this->lineEdit_Damping->setValidator(this->number_validator);
	this->lineEdit_dt->setValidator(this->number_validator_unsigned);
	//		GNEB
	this->lineEdit_gneb_springconstant->setValidator(this->number_validator);

	// Visualisation
	//		Arrows
	this->lineEdit_arrows_lod->setValidator(this->number_validator_int_unsigned);
	//		Isovalue
	this->lineEdit_isovalue->setValidator(this->number_validator);
	//		Camera
	this->lineEdit_camera_pos_x->setValidator(this->number_validator);
	this->lineEdit_camera_pos_y->setValidator(this->number_validator);
	this->lineEdit_camera_pos_z->setValidator(this->number_validator);
	this->lineEdit_camera_focus_x->setValidator(this->number_validator);
	this->lineEdit_camera_focus_y->setValidator(this->number_validator);
	this->lineEdit_camera_focus_z->setValidator(this->number_validator);
}
