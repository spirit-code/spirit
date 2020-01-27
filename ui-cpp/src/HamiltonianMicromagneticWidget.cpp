#include <QtWidgets>
#include <QDialog>

#include "HamiltonianMicromagneticWidget.hpp"

#include <Spirit/System.h>
#include <Spirit/Geometry.h>
#include <Spirit/Chain.h>
#include <Spirit/Log.h>
#include <Spirit/Hamiltonian.h>

#define Exception_Division_by_zero 6666
template <typename T>
void normalize(T v[3])
{
    T len = 0.0;
    for (int i = 0; i < 3; ++i) len += std::pow(v[i], 2);
    if (len == 0.0) throw Exception_Division_by_zero;
    for (int i = 0; i < 3; ++i) v[i] /= std::sqrt(len);
}

HamiltonianMicromagneticWidget::HamiltonianMicromagneticWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
{
    this->state = state;
    this->spinWidget = spinWidget;

    // Setup User Interface
    this->setupUi(this);

    // Setup the validators for the various input fields
    //this->Setup_Input_Validators();
    //this->Setup_comboBox();
    // Load variables from SpinWidget and State
    //this->updateData();

    // Connect signals and slots
    //this->Setup_Slots();
}

void HamiltonianMicromagneticWidget::setData()
{
	this->Setup_Input_Validators();
	int region_num;
	Hamiltonian_Get_Regions(state.get(), &region_num);
	for (int i=0;i<region_num;i++){
	QString inp=QString("region %1").arg(i);
	this->comboBox_region->addItem(inp,i);
	}
	this->updateData();
	this->Setup_Slots();
}

void HamiltonianMicromagneticWidget::updateData()
{
    float d, vd[3], jij[100], dij[100];
    float tensor[9];
    int region_num;
    int ddi_method, ddi_n_periodic_images[3];
    bool pb_zero_padding;
    int n_neigh_shells_exchange, n_neigh_shells_dmi, dm_chirality;
    int n_basis_atoms = Geometry_Get_N_Cell_Atoms(state.get());
    std::vector<float> mu_s(n_basis_atoms);
    // Boundary conditions
    bool boundary_conditions[3];
    Hamiltonian_Get_Boundary_Conditions(state.get(), boundary_conditions);
    this->checkBox_aniso_periodical_a->setChecked(boundary_conditions[0]);
    this->checkBox_aniso_periodical_b->setChecked(boundary_conditions[1]);
    this->checkBox_aniso_periodical_c->setChecked(boundary_conditions[2]);
    // Ms
    Hamiltonian_Get_Ms(state.get(), &d,this->comboBox_region->currentIndex());
    this->lineEdit_Ms->setText(QString::number(d));

    // cell_sizes
	Hamiltonian_Get_Cell_Sizes(state.get(), vd);
	this->lineEdit_cell_sizes_x->setText(QString::number(vd[0]));
	this->lineEdit_cell_sizes_y->setText(QString::number(vd[1]));
	this->lineEdit_cell_sizes_z->setText(QString::number(vd[2]));

    // External magnetic field
    Hamiltonian_Get_Field_Regions(state.get(), &d, vd,this->comboBox_region->currentIndex());
    this->lineEdit_extH_aniso->setText(QString::number(d));
    this->lineEdit_extHx_aniso->setText(QString::number(vd[0]));
    this->lineEdit_extHy_aniso->setText(QString::number(vd[1]));
    this->lineEdit_extHz_aniso->setText(QString::number(vd[2]));
    if (d > 0.0) this->checkBox_extH_aniso->setChecked(true);

    // Anisotropy
    Hamiltonian_Get_Anisotropy_Regions(state.get(), &d, vd,this->comboBox_region->currentIndex());
    this->lineEdit_ani_aniso->setText(QString::number(d));
    this->lineEdit_anix_aniso->setText(QString::number(vd[0]));
    this->lineEdit_aniy_aniso->setText(QString::number(vd[1]));
    this->lineEdit_aniz_aniso->setText(QString::number(vd[2]));
    if (d > 0.0) this->checkBox_ani_aniso->setChecked(true);

    Hamiltonian_Get_Exchange_Tensor(state.get(), &d,this->comboBox_region->currentIndex());
	this->lineEdit_exch_00->setText(QString::number(d));
	this->checkBox_exchange->setChecked(true);

	Hamiltonian_Get_DMI_Tensor(state.get(), &d,this->comboBox_region->currentIndex());
	this->lineEdit_dmi_00->setText(QString::number(d));
	this->checkBox_dmi->setChecked(true);
    // // Exchange interaction (shells)
    // Hamiltonian_Get_Exchange_Shells(state.get(), &n_neigh_shells_exchange, jij);
    // if (n_neigh_shells_exchange > 0) this->checkBox_exchange->setChecked(true);
    // else this->checkBox_exchange->setChecked(false);
    // this->spinBox_nshells_exchange->setValue(n_neigh_shells_exchange);
    // this->set_nshells_exchange();
    // for (int i = 0; i < n_neigh_shells_exchange; ++i) this->exchange_shells[i]->setValue(jij[i]);

    // // DMI
    // Hamiltonian_Get_DMI_Shells(state.get(), &n_neigh_shells_dmi, dij, &dm_chirality);
    // int index = 0;
    // if( dm_chirality == -1 ) index = 1;
    // else if( dm_chirality == 2 ) index = 2;
    // else if( dm_chirality == -2 ) index = 3;
    // this->comboBox_dmi_chirality->setCurrentIndex(index);
    // if (n_neigh_shells_dmi > 0) this->checkBox_dmi->setChecked(true);
    // this->spinBox_nshells_dmi->setValue(n_neigh_shells_dmi);
    // this->set_nshells_dmi();
    // for (int i = 0; i < n_neigh_shells_dmi; ++i) this->dmi_shells[i]->setValue(dij[i]);

    // DDI
    Hamiltonian_Get_DDI(state.get(), &ddi_method, ddi_n_periodic_images, &d, &pb_zero_padding);
    this->checkBox_ddi->setChecked( ddi_method != SPIRIT_DDI_METHOD_NONE );
    if( ddi_method == SPIRIT_DDI_METHOD_NONE )
        this->comboBox_ddi_method->setCurrentIndex(0);
    else if( ddi_method == SPIRIT_DDI_METHOD_FFT )
        this->comboBox_ddi_method->setCurrentIndex(0);
    else if( ddi_method == SPIRIT_DDI_METHOD_FMM )
        this->comboBox_ddi_method->setCurrentIndex(1);
    else if( ddi_method == SPIRIT_DDI_METHOD_CUTOFF )
        this->comboBox_ddi_method->setCurrentIndex(2);
    this->spinBox_ddi_n_periodic_a->setValue(ddi_n_periodic_images[0]);
    this->spinBox_ddi_n_periodic_b->setValue(ddi_n_periodic_images[1]);
    this->spinBox_ddi_n_periodic_c->setValue(ddi_n_periodic_images[2]);
    this->doubleSpinBox_ddi_radius->setValue(d);
}
void HamiltonianMicromagneticWidget::clicked_region()
{
	this->updateData();
}
void HamiltonianMicromagneticWidget::clicked_change_hamiltonian()
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


// -----------------------------------------------------------------------------------
// -------------------------- Setters ------------------------------------------------
// -----------------------------------------------------------------------------------


void HamiltonianMicromagneticWidget::set_boundary_conditions()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
    {
        // Boundary conditions
        bool boundary_conditions[3];
        boundary_conditions[0] = this->checkBox_aniso_periodical_a->isChecked();
        boundary_conditions[1] = this->checkBox_aniso_periodical_b->isChecked();
        boundary_conditions[2] = this->checkBox_aniso_periodical_c->isChecked();
        Hamiltonian_Set_Boundary_Conditions(state.get(), boundary_conditions, idx_image);
    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
    this->spinWidget->updateBoundingBoxIndicators();
}

void HamiltonianMicromagneticWidget::set_cell_sizes()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
    {
        float vd[3];

        // External magnetic field
        //		magnitude
        //		normal
        vd[0] = lineEdit_cell_sizes_x->text().toFloat();
        vd[1] = lineEdit_cell_sizes_y->text().toFloat();
        vd[2] = lineEdit_cell_sizes_z->text().toFloat();

        Hamiltonian_Set_Cell_Sizes(state.get(), vd, idx_image);
    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}

void HamiltonianMicromagneticWidget::set_Ms()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
    {
        float d;

        // External magnetic field
        //		magnitude
        //		normal
        d = lineEdit_Ms->text().toFloat();

        Hamiltonian_Set_Ms(state.get(), d, this->comboBox_region->currentIndex(), idx_image);
    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}
void HamiltonianMicromagneticWidget::set_external_field()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
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
        Hamiltonian_Set_Field_Regions(state.get(), d, vd, this->comboBox_region->currentIndex(), idx_image);
    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}


void HamiltonianMicromagneticWidget::set_anisotropy()
{
	//printf("sdasd\n");
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
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
        Hamiltonian_Set_Anisotropy_Regions(state.get(), d, vd, this->comboBox_region->currentIndex(), idx_image);
    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}

void HamiltonianMicromagneticWidget::set_exchange()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
    {
        if (this->checkBox_exchange->isChecked())
        {
            float tensor;
            tensor=lineEdit_exch_00->text().toFloat();
            Hamiltonian_Set_Exchange_Tensor(state.get(), tensor, this->comboBox_region->currentIndex(), idx_image);
        } else{
        	float tensor=0;
        	Hamiltonian_Set_Exchange_Tensor(state.get(), tensor, this->comboBox_region->currentIndex(), idx_image);
        }

    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}

void HamiltonianMicromagneticWidget::set_dmi()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this](int idx_image) -> void
    {
        if (this->checkBox_dmi->isChecked())
        {
            float tensor;
            tensor=lineEdit_dmi_00->text().toFloat();
            Hamiltonian_Set_DMI_Tensor(state.get(), tensor, this->comboBox_region->currentIndex(), idx_image);
        } else{
        	float tensor=0;
			Hamiltonian_Set_DMI_Tensor(state.get(), tensor, this->comboBox_region->currentIndex(), idx_image);
        }


    };

    if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()));
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int i = 0; i<Chain_Get_NOI(state.get()); ++i)
        {
            apply(i);
        }
    }
    else if (this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img);
        }
    }
}
// -----------------------------------------------------------------------------------
// --------------------------------- Setup -------------------------------------------
// -----------------------------------------------------------------------------------


void HamiltonianMicromagneticWidget::Setup_Input_Validators()
{
    // We use a regular expression (regex) to filter the input into the lineEdits
    QRegularExpression re("[+|-]?[\\d]*[\\.]?[\\d]*((e\\+)?|(e\\-)?|(e)?)[\\d]*");
    this->number_validator = new QRegularExpressionValidator(re);
    QRegularExpression re2("[\\d]*[\\.]?[\\d]*");
    this->number_validator_unsigned = new QRegularExpressionValidator(re2);
    QRegularExpression re3("[+|-]?[\\d]*");
    this->number_validator_int = new QRegularExpressionValidator(re3);
    QRegularExpression re4("[\\d]*");
    this->number_validator_int_unsigned = new QRegularExpressionValidator(re4);
    //      Ms
    this->lineEdit_cell_sizes_x->setValidator(this->number_validator);
    this->lineEdit_cell_sizes_y->setValidator(this->number_validator);
    this->lineEdit_cell_sizes_z->setValidator(this->number_validator);
    //      Ms
    this->lineEdit_Ms->setValidator(this->number_validator);
    //      external field
    this->lineEdit_extH_aniso->setValidator(this->number_validator);
    this->lineEdit_extHx_aniso->setValidator(this->number_validator);
    this->lineEdit_extHy_aniso->setValidator(this->number_validator);
    this->lineEdit_extHz_aniso->setValidator(this->number_validator);
    //      anisotropy
    this->lineEdit_ani_aniso->setValidator(this->number_validator);
    this->lineEdit_anix_aniso->setValidator(this->number_validator);
    this->lineEdit_aniy_aniso->setValidator(this->number_validator);
    this->lineEdit_aniz_aniso->setValidator(this->number_validator);
    // 		exchange
    this->lineEdit_exch_00->setValidator(this->number_validator);

    //		DMI
    this->lineEdit_dmi_00->setValidator(this->number_validator);

}

void HamiltonianMicromagneticWidget::Setup_Slots()
{
    connect(this->pushButton_changeHamiltonian, SIGNAL(clicked()), this, SLOT(clicked_change_hamiltonian()));
    // Boundary Conditions
    connect(this->checkBox_aniso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
    connect(this->checkBox_aniso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
    connect(this->checkBox_aniso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
    // Ms
    connect(this->lineEdit_Ms, SIGNAL(returnPressed()), this, SLOT(set_Ms()));
    connect(this->comboBox_region, SIGNAL(currentIndexChanged(int)), this, SLOT(clicked_region()));
    // Cell sizes
	connect(this->lineEdit_cell_sizes_x, SIGNAL(returnPressed()), this, SLOT(set_cell_sizes()));
	connect(this->lineEdit_cell_sizes_y, SIGNAL(returnPressed()), this, SLOT(set_cell_sizes()));
	connect(this->lineEdit_cell_sizes_z, SIGNAL(returnPressed()), this, SLOT(set_cell_sizes()));
	// External Field
    connect(this->checkBox_extH_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_external_field()));
	connect(this->lineEdit_extH_aniso, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHx_aniso, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHy_aniso, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	connect(this->lineEdit_extHz_aniso, SIGNAL(returnPressed()), this, SLOT(set_external_field()));
	// Anisotropy
	connect(this->checkBox_ani_aniso, SIGNAL(stateChanged(int)), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_ani_aniso, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_anix_aniso, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_aniy_aniso, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	connect(this->lineEdit_aniz_aniso, SIGNAL(returnPressed()), this, SLOT(set_anisotropy()));
	// Exchange
	connect(this->checkBox_exchange, SIGNAL(stateChanged(int)), this, SLOT(set_exchange()));
	connect(this->lineEdit_exch_00, SIGNAL(returnPressed()), this, SLOT(set_exchange()));

	// DMI
	connect(this->checkBox_dmi, SIGNAL(stateChanged(int)), this, SLOT(set_dmi()));
	connect(this->lineEdit_dmi_00, SIGNAL(returnPressed()), this, SLOT(set_dmi()));

}
