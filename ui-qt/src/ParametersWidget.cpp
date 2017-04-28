#include <QtWidgets>

#include "ParametersWidget.hpp"

#include <Spirit/Parameters.h>
#include <Spirit/System.h>
#include <Spirit/Chain.h>
#include <Spirit/Collection.h>
#include <Spirit/Log.h>
#include <Spirit/Exception.h>

// Small function for normalization of vectors
template <typename T>
void normalize(T v[3])
{
	T len = 0.0;
	for (int i = 0; i < 3; ++i) len += std::pow(v[i], 2);
	if (len == 0.0) throw Exception_Division_by_zero;
	for (int i = 0; i < 3; ++i) v[i] /= std::sqrt(len);
}

ParametersWidget::ParametersWidget(std::shared_ptr<State> state)
{
	this->state = state;
	//this->spinWidget = spinWidget;
	//this->settingsWidget = settingsWidget;

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
	this->Setup_Parameters_Slots();
}

void ParametersWidget::updateData()
{
	this->Load_Parameters_Contents();
}


void ParametersWidget::Load_Parameters_Contents()
{
	float d, vd[3];
	int image_type;
	int i1, i2;

	//		LLG
	// LLG Damping
	d = Parameters_Get_LLG_Damping(state.get());
	this->lineEdit_Damping->setText(QString::number(d));
	// Converto to PicoSeconds
	d = Parameters_Get_LLG_Time_Step(state.get());
	this->lineEdit_dt->setText(QString::number(d));
	// LLG Iteration Params
	Parameters_Get_LLG_N_Iterations(state.get(), &i1, &i2);
	this->lineEdit_llg_n_iterations->setText(QString::number(i1));
	this->lineEdit_llg_log_steps->setText(QString::number(i2));
	// Spin polarized current
	Parameters_Get_LLG_STT(state.get(), &d, vd);
	this->doubleSpinBox_llg_stt_magnitude->setValue(d);
	this->doubleSpinBox_llg_stt_polarisation_x->setValue(vd[0]);
	this->doubleSpinBox_llg_stt_polarisation_y->setValue(vd[1]);
	this->doubleSpinBox_llg_stt_polarisation_z->setValue(vd[2]);
	if (d > 0.0) this->checkBox_llg_stt->setChecked(true);
	// Temperature
	d = Parameters_Get_LLG_Temperature(state.get());
	this->doubleSpinBox_llg_temperature->setValue(d);
	if (d > 0.0) this->checkBox_llg_temperature->setChecked(true);

	//		MC
	d = Parameters_Get_MC_Temperature(state.get());
	this->doubleSpinBox_mc_temperature->setValue(d);
	if (d > 0.0) this->checkBox_mc_temperature->setChecked(true);
	d = Parameters_Get_MC_Acceptance_Ratio(state.get());
	this->doubleSpinBox_mc_acceptance->setValue(d);

	//		GNEB
	// GNEB Interation Params
	Parameters_Get_GNEB_N_Iterations(state.get(), &i1, &i2);
	this->lineEdit_gneb_n_iterations->setText(QString::number(i1));
	this->lineEdit_gneb_log_steps->setText(QString::number(i2));

	// GNEB Spring Constant
	d = Parameters_Get_GNEB_Spring_Constant(state.get());
	this->lineEdit_gneb_springconstant->setText(QString::number(d));

	// Normal/Climbing/Falling image radioButtons
	image_type = Parameters_Get_GNEB_Climbing_Falling(state.get());
	if (image_type == 0)
		this->radioButton_Normal->setChecked(true);
	else if (image_type == 1)
		this->radioButton_ClimbingImage->setChecked(true);
	else if (image_type == 2)
		this->radioButton_FallingImage->setChecked(true);
	else if (image_type == 3)
		this->radioButton_Stationary->setChecked(true);
}


void ParametersWidget::set_parameters()
{
	// Closure to set the parameters of a specific spin system
	auto apply = [this](int idx_image, int idx_chain) -> void
	{
		float d, vd[3];
		int i1, i2;

		//		LLG
		// Time step [ps]
		// dt = time_step [ps] * 10^-12 * gyromagnetic raio / mu_B  { / (1+damping^2)} <- not implemented
		d = this->lineEdit_dt->text().toFloat();
		Parameters_Set_LLG_Time_Step(this->state.get(), d, idx_image, idx_chain);
		
		// Damping
		d = this->lineEdit_Damping->text().toFloat();
		Parameters_Set_LLG_Damping(this->state.get(), d);
		// n iterations
		i1 = this->lineEdit_llg_n_iterations->text().toInt();
		i2 = this->lineEdit_llg_log_steps->text().toInt();
		Parameters_Set_LLG_N_Iterations(state.get(), i1, i2);
		i1 = this->lineEdit_gneb_n_iterations->text().toInt();
		i2 = this->lineEdit_gneb_log_steps->text().toInt();
		Parameters_Set_GNEB_N_Iterations(state.get(), i1, i2);
		// Spin polarised current
		if (this->checkBox_llg_stt->isChecked())
			d = this->doubleSpinBox_llg_stt_magnitude->value();
		else
			d = 0.0;
		vd[0] = doubleSpinBox_llg_stt_polarisation_x->value();
		vd[1] = doubleSpinBox_llg_stt_polarisation_y->value();
		vd[2] = doubleSpinBox_llg_stt_polarisation_z->value();
		try {
			normalize(vd);
		}
		catch (int ex) {
			if (ex == Exception_Division_by_zero) {
				vd[0] = 0.0;
				vd[1] = 0.0;
				vd[2] = 1.0;
				Log_Send(state.get(), Log_Level_Warning, Log_Sender_UI, "s_c_vec = {0,0,0} replaced by {0,0,1}");
				doubleSpinBox_llg_stt_polarisation_x->setValue(0.0);
				doubleSpinBox_llg_stt_polarisation_y->setValue(0.0);
				doubleSpinBox_llg_stt_polarisation_z->setValue(1.0);
			}
			else { throw(ex); }
		}
		Parameters_Set_LLG_STT(state.get(), d, vd, idx_image, idx_chain);
		// Temperature
		if (this->checkBox_llg_temperature->isChecked())
			d = this->doubleSpinBox_llg_temperature->value();
		else
			d = 0.0;
		Parameters_Set_LLG_Temperature(state.get(), d, idx_image, idx_chain);

		//		MC
		if (this->checkBox_mc_temperature->isChecked())
			d = this->doubleSpinBox_mc_temperature->value();
		else
			d = 0.0;
		Parameters_Set_MC_Temperature(state.get(), d, idx_image, idx_chain);
		d = this->doubleSpinBox_mc_acceptance->value();
		Parameters_Set_MC_Acceptance_Ratio(state.get(), d, idx_image, idx_chain);

		//		GNEB
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


void ParametersWidget::Setup_Parameters_Slots()
{
	// Iteration params
	connect(this->lineEdit_llg_n_iterations, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_llg_log_steps, SIGNAL(returnPressed()), this, SLOT(set_parameters()));

	//		LLG
	// Temperature
	connect(this->checkBox_llg_temperature, SIGNAL(stateChanged(int)), this, SLOT(set_parameters()));
	connect(this->doubleSpinBox_llg_temperature, SIGNAL(editingFinished()), this, SLOT(set_parameters()));
	// STT
	connect(this->checkBox_llg_stt, SIGNAL(stateChanged(int)), this, SLOT(set_parameters()));
	connect(this->doubleSpinBox_llg_stt_magnitude, SIGNAL(editingFinished()), this, SLOT(set_parameters()));
	connect(this->doubleSpinBox_llg_stt_polarisation_x, SIGNAL(editingFinished()), this, SLOT(set_parameters()));
	connect(this->doubleSpinBox_llg_stt_polarisation_y, SIGNAL(editingFinished()), this, SLOT(set_parameters()));
	connect(this->doubleSpinBox_llg_stt_polarisation_z, SIGNAL(editingFinished()), this, SLOT(set_parameters()));
	// Damping
	connect(this->lineEdit_Damping, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_dt, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// iteration params
	connect(this->lineEdit_llg_n_iterations, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	connect(this->lineEdit_llg_log_steps, SIGNAL(returnPressed()), this, SLOT(set_parameters()));

	//		GNEB
	// Spring Constant
	connect(this->lineEdit_gneb_springconstant, SIGNAL(returnPressed()), this, SLOT(set_parameters()));
	// Normal/Climbing/Falling image radioButtons
	connect(this->radioButton_Normal, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_ClimbingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_FallingImage, SIGNAL(clicked()), this, SLOT(set_parameters()));
	connect(this->radioButton_Stationary, SIGNAL(clicked()), this, SLOT(set_parameters()));
}


void ParametersWidget::Setup_Input_Validators()
{
	//		LLG
	this->lineEdit_Damping->setValidator(this->number_validator_unsigned);
	this->lineEdit_dt->setValidator(this->number_validator_unsigned);
	//		GNEB
	this->lineEdit_gneb_springconstant->setValidator(this->number_validator);
}