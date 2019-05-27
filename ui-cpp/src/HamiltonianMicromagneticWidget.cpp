#include <QtWidgets>
#include <QDialog>

#include "HamiltonianMicromagneticWidget.hpp"

#include <Spirit/System.h>
#include <Spirit/Geometry.h>
#include <Spirit/Chain.h>
#include <Spirit/Log.h>
#include <Spirit/Hamiltonian.h>


HamiltonianMicromagneticWidget::HamiltonianMicromagneticWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
{
    this->state = state;
    this->spinWidget = spinWidget;

    // Setup User Interface
    this->setupUi(this);

    // Setup the validators for the various input fields
    this->Setup_Input_Validators();

    // Load variables from SpinWidget and State
    this->updateData();

    // Connect signals and slots
    this->Setup_Slots();
}

void HamiltonianMicromagneticWidget::updateData()
{
    float d, vd[3], jij[100], dij[100];
    int ddi_method, ddi_n_periodic_images[3];
    int n_neigh_shells_exchange, n_neigh_shells_dmi, dm_chirality;
    int n_basis_atoms = Geometry_Get_N_Cell_Atoms(state.get());
    std::vector<float> mu_s(n_basis_atoms);

    // Boundary conditions
    bool boundary_conditions[3];
    Hamiltonian_Get_Boundary_Conditions(state.get(), boundary_conditions);
    this->checkBox_aniso_periodical_a->setChecked(boundary_conditions[0]);
    this->checkBox_aniso_periodical_b->setChecked(boundary_conditions[1]);
    this->checkBox_aniso_periodical_c->setChecked(boundary_conditions[2]);

    // mu_s
    Geometry_Get_mu_s(state.get(), mu_s.data());
    this->lineEdit_muSpin_aniso->setText(QString::number(mu_s[0]));

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
    Hamiltonian_Get_DDI(state.get(), &ddi_method, ddi_n_periodic_images, &d);
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


// -----------------------------------------------------------------------------------
// --------------------------------- Setup -------------------------------------------
// -----------------------------------------------------------------------------------


void HamiltonianMicromagneticWidget::Setup_Input_Validators()
{
    // We use a regular expression (regex) to filter the input into the lineEdits
    QRegularExpression re("[+|-]?[\\d]*[\\.]?[\\d]*");
    this->number_validator = new QRegularExpressionValidator(re);
    QRegularExpression re2("[\\d]*[\\.]?[\\d]*");
    this->number_validator_unsigned = new QRegularExpressionValidator(re2);
    QRegularExpression re3("[+|-]?[\\d]*");
    this->number_validator_int = new QRegularExpressionValidator(re3);
    QRegularExpression re4("[\\d]*");
    this->number_validator_int_unsigned = new QRegularExpressionValidator(re4);

    //      mu_s
    this->lineEdit_muSpin_aniso->setValidator(this->number_validator);
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
}

void HamiltonianMicromagneticWidget::Setup_Slots()
{
    connect(this->pushButton_changeHamiltonian, SIGNAL(clicked()), this, SLOT(clicked_change_hamiltonian()));
    // Boundary Conditions
    connect(this->checkBox_aniso_periodical_a, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
    connect(this->checkBox_aniso_periodical_b, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
    connect(this->checkBox_aniso_periodical_c, SIGNAL(stateChanged(int)), this, SLOT(set_boundary_conditions()));
}