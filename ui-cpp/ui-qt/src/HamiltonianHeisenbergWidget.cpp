#include "HamiltonianHeisenbergWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Log.h>
#include <Spirit/System.h>

// Small function for normalization of vectors
#define Exception_Division_by_zero 6666
template<typename T>
void normalize( T v[3] )
{
    T len = 0.0;
    for( int i = 0; i < 3; ++i )
        len += std::pow( v[i], 2 );
    if( len == 0.0 )
        throw Exception_Division_by_zero;
    for( int i = 0; i < 3; ++i )
        v[i] /= std::sqrt( len );
}

HamiltonianHeisenbergWidget::HamiltonianHeisenbergWidget( std::shared_ptr<State> state, SpinWidget * spinWidget )
{
    this->state      = state;
    this->spinWidget = spinWidget;

    // Setup User Interface
    this->setupUi( this );

    // We use a regular expression (regex) to filter the input into the lineEdits
    QRegularExpression re( "[+|-]?[\\d]*[\\.]?[\\d]*" );
    this->number_validator = new QRegularExpressionValidator( re );
    QRegularExpression re2( "[\\d]*[\\.]?[\\d]*" );
    this->number_validator_unsigned = new QRegularExpressionValidator( re2 );
    QRegularExpression re3( "[+|-]?[\\d]*" );
    this->number_validator_int = new QRegularExpressionValidator( re3 );
    QRegularExpression re4( "[\\d]*" );
    this->number_validator_int_unsigned = new QRegularExpressionValidator( re4 );
    // Setup the validators for the various input fields
    this->Setup_Input_Validators();

    // Load variables from SpinWidget and State
    this->updateData();

    // Connect signals and slots
    this->Setup_Slots();
}

void HamiltonianHeisenbergWidget::updateData()
{
    Load_Contents();
}

void HamiltonianHeisenbergWidget::Load_Contents()
{
    float d, vd[3], jij[100], dij[100];
    int ddi_method, ddi_n_periodic_images[3];
    bool pb_zero_padding;
    int n_neigh_shells_exchange, n_neigh_shells_dmi, dm_chirality;
    int n_basis_atoms = Geometry_Get_N_Cell_Atoms( state.get() );
    std::vector<float> mu_s( n_basis_atoms );

    // Boundary conditions
    bool boundary_conditions[3];
    Hamiltonian_Get_Boundary_Conditions( state.get(), boundary_conditions );
    this->checkBox_aniso_periodical_a->setChecked( boundary_conditions[0] );
    this->checkBox_aniso_periodical_b->setChecked( boundary_conditions[1] );
    this->checkBox_aniso_periodical_c->setChecked( boundary_conditions[2] );

    // mu_s
    Geometry_Get_mu_s( state.get(), mu_s.data() );
    this->lineEdit_muSpin_aniso->setText( QString::number( mu_s[0] ) );

    // External magnetic field
    Hamiltonian_Get_Field( state.get(), &d, vd );
    this->lineEdit_extH_aniso->setText( QString::number( d ) );
    this->lineEdit_extHx_aniso->setText( QString::number( vd[0] ) );
    this->lineEdit_extHy_aniso->setText( QString::number( vd[1] ) );
    this->lineEdit_extHz_aniso->setText( QString::number( vd[2] ) );
    if( std::abs( d ) > 0.0 )
        this->checkBox_extH_aniso->setChecked( true );

    // Anisotropy
    Hamiltonian_Get_Anisotropy( state.get(), &d, vd );
    this->lineEdit_ani_aniso->setText( QString::number( d ) );
    this->lineEdit_anix_aniso->setText( QString::number( vd[0] ) );
    this->lineEdit_aniy_aniso->setText( QString::number( vd[1] ) );
    this->lineEdit_aniz_aniso->setText( QString::number( vd[2] ) );
    if( d > 0.0 )
        this->checkBox_ani_aniso->setChecked( true );

    // Exchange interaction (shells)
    Hamiltonian_Get_Exchange_Shells( state.get(), &n_neigh_shells_exchange, jij );
    if( n_neigh_shells_exchange > 0 )
        this->checkBox_exchange->setChecked( true );
    else
        this->checkBox_exchange->setChecked( false );
    this->spinBox_nshells_exchange->setValue( n_neigh_shells_exchange );
    this->set_nshells_exchange();
    for( int i = 0; i < n_neigh_shells_exchange; ++i )
        this->exchange_shells[i]->setValue( jij[i] );

    // DMI
    Hamiltonian_Get_DMI_Shells( state.get(), &n_neigh_shells_dmi, dij, &dm_chirality );
    int index = 0;
    if( dm_chirality == -1 )
        index = 1;
    else if( dm_chirality == 2 )
        index = 2;
    else if( dm_chirality == -2 )
        index = 3;
    this->comboBox_dmi_chirality->setCurrentIndex( index );
    if( n_neigh_shells_dmi > 0 )
        this->checkBox_dmi->setChecked( true );
    this->spinBox_nshells_dmi->setValue( n_neigh_shells_dmi );
    this->set_nshells_dmi();
    for( int i = 0; i < n_neigh_shells_dmi; ++i )
        this->dmi_shells[i]->setValue( dij[i] );

    // DDI
    Hamiltonian_Get_DDI( state.get(), &ddi_method, ddi_n_periodic_images, &d, &pb_zero_padding );
    this->checkBox_ddi->setChecked( ddi_method != SPIRIT_DDI_METHOD_NONE );
    if( ddi_method == SPIRIT_DDI_METHOD_NONE )
        this->comboBox_ddi_method->setCurrentIndex( 0 );
    else if( ddi_method == SPIRIT_DDI_METHOD_FFT )
        this->comboBox_ddi_method->setCurrentIndex( 0 );
    else if( ddi_method == SPIRIT_DDI_METHOD_FMM )
        this->comboBox_ddi_method->setCurrentIndex( 1 );
    else if( ddi_method == SPIRIT_DDI_METHOD_CUTOFF )
        this->comboBox_ddi_method->setCurrentIndex( 2 );
    this->spinBox_ddi_n_periodic_a->setValue( ddi_n_periodic_images[0] );
    this->spinBox_ddi_n_periodic_b->setValue( ddi_n_periodic_images[1] );
    this->spinBox_ddi_n_periodic_c->setValue( ddi_n_periodic_images[2] );
    this->doubleSpinBox_ddi_radius->setValue( d );
    this->checkBox_ddi_pb_zero_padding->setChecked( pb_zero_padding );
}

// -----------------------------------------------------------------------------------
// -------------------------- Setters ------------------------------------------------
// -----------------------------------------------------------------------------------

void HamiltonianHeisenbergWidget::set_boundary_conditions()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        // Boundary conditions
        bool boundary_conditions[3];
        boundary_conditions[0] = this->checkBox_aniso_periodical_a->isChecked();
        boundary_conditions[1] = this->checkBox_aniso_periodical_b->isChecked();
        boundary_conditions[2] = this->checkBox_aniso_periodical_c->isChecked();
        Hamiltonian_Set_Boundary_Conditions( state.get(), boundary_conditions, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
    this->spinWidget->updateBoundingBoxIndicators();
}

void HamiltonianHeisenbergWidget::set_mu_s()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        // mu_s
        float mu_s = this->lineEdit_muSpin_aniso->text().toFloat();
        Geometry_Set_mu_s( state.get(), mu_s, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_external_field()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        float d, vd[3];

        // External magnetic field
        //      magnitude
        if( this->checkBox_extH_aniso->isChecked() )
            d = this->lineEdit_extH_aniso->text().toFloat();
        else
            d = 0.0;
        //      normal
        vd[0] = lineEdit_extHx_aniso->text().toFloat();
        vd[1] = lineEdit_extHy_aniso->text().toFloat();
        vd[2] = lineEdit_extHz_aniso->text().toFloat();
        try
        {
            normalize( vd );
        }
        catch( int ex )
        {
            if( ex == Exception_Division_by_zero )
            {
                vd[0] = 0.0;
                vd[1] = 0.0;
                vd[2] = 1.0;
                Log_Send( state.get(), Log_Level_Warning, Log_Sender_UI, "B_vec = {0,0,0} replaced by {0,0,1}" );
                lineEdit_extHx_aniso->setText( QString::number( 0.0 ) );
                lineEdit_extHy_aniso->setText( QString::number( 0.0 ) );
                lineEdit_extHz_aniso->setText( QString::number( 1.0 ) );
            }
            else
            {
                throw( ex );
            }
        }
        Hamiltonian_Set_Field( state.get(), d, vd, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_anisotropy()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        float d, vd[3];

        // Anisotropy
        //      magnitude
        if( this->checkBox_ani_aniso->isChecked() )
            d = this->lineEdit_ani_aniso->text().toFloat();
        else
            d = 0.0;
        //      normal
        vd[0] = lineEdit_anix_aniso->text().toFloat();
        vd[1] = lineEdit_aniy_aniso->text().toFloat();
        vd[2] = lineEdit_aniz_aniso->text().toFloat();
        try
        {
            normalize( vd );
        }
        catch( int ex )
        {
            if( ex == Exception_Division_by_zero )
            {
                vd[0] = 0.0;
                vd[1] = 0.0;
                vd[2] = 1.0;
                Log_Send( state.get(), Log_Level_Warning, Log_Sender_UI, "ani_vec = {0,0,0} replaced by {0,0,1}" );
                lineEdit_anix_aniso->setText( QString::number( 0.0 ) );
                lineEdit_aniy_aniso->setText( QString::number( 0.0 ) );
                lineEdit_aniz_aniso->setText( QString::number( 1.0 ) );
            }
            else
            {
                throw( ex );
            }
        }
        Hamiltonian_Set_Anisotropy( state.get(), d, vd, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_nshells_exchange()
{
    // The desired number of shells
    int n_shells = this->spinBox_nshells_exchange->value();
    // The current number of shells
    int n_shells_current = this->exchange_shells.size();
    // If reduction remove widgets and set exchange
    if( n_shells < n_shells_current )
    {
        for( int n = n_shells_current; n > n_shells; --n )
        {
            this->exchange_shells.back()->close();
            this->exchange_shells.pop_back();
        }
    }
    // If increase add widgets, connect to slots and do nothing
    else
    {
        for( int n = n_shells_current; n < n_shells; ++n )
        {
            auto x = new QDoubleSpinBox();
            x->setDecimals( 4 );
            x->setRange( -1000, 1000 );
            this->exchange_shells.push_back( x );
            this->gridLayout_exchange->addWidget( x );
            connect( x, SIGNAL( editingFinished() ), this, SLOT( set_exchange() ) );
        }
    }
}

void HamiltonianHeisenbergWidget::set_exchange()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        if( this->checkBox_exchange->isChecked() )
        {
            int n_shells = this->exchange_shells.size();
            std::vector<float> Jij( n_shells );
            for( int i = 0; i < n_shells; ++i )
                Jij[i] = this->exchange_shells[i]->value();
            Hamiltonian_Set_Exchange( state.get(), n_shells, Jij.data(), idx_image );
        }
        else
            Hamiltonian_Set_Exchange( state.get(), 0, nullptr, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_nshells_dmi()
{
    // The desired number of shells
    int n_shells = this->spinBox_nshells_dmi->value();
    // The current number of shells
    int n_shells_current = this->dmi_shells.size();
    // If reduction remove widgets and set dmi
    if( n_shells < n_shells_current )
    {
        for( int n = n_shells_current; n > n_shells; --n )
        {
            this->dmi_shells.back()->close();
            this->dmi_shells.pop_back();
        }
    }
    // If increase add widgets, connect to slots and do nothing
    else
    {
        for( int n = n_shells_current; n < n_shells; ++n )
        {
            auto x = new QDoubleSpinBox();
            x->setDecimals( 4 );
            x->setRange( -1000, 1000 );
            this->dmi_shells.push_back( x );
            this->gridLayout_dmi->addWidget( x );
            connect( x, SIGNAL( editingFinished() ), this, SLOT( set_dmi() ) );
        }
    }
}

void HamiltonianHeisenbergWidget::set_dmi()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        if( this->checkBox_dmi->isChecked() )
        {
            int n_shells = this->dmi_shells.size();
            std::vector<float> Dij( n_shells );
            for( int i = 0; i < n_shells; ++i )
                Dij[i] = this->dmi_shells[i]->value();

            int chirality = 1;
            if( this->comboBox_dmi_chirality->currentIndex() == 1 )
                chirality = -1;
            else if( this->comboBox_dmi_chirality->currentIndex() == 2 )
                chirality = 2;
            else if( this->comboBox_dmi_chirality->currentIndex() == 3 )
                chirality = -2;

            Hamiltonian_Set_DMI( state.get(), n_shells, Dij.data(), chirality, idx_image );
        }
        else
            Hamiltonian_Set_DMI( state.get(), 0, nullptr, 1, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_ddi()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        int method = SPIRIT_DDI_METHOD_NONE;
        if( this->checkBox_ddi->isChecked() )
        {
            if( this->comboBox_ddi_method->currentIndex() == 0 )
                method = SPIRIT_DDI_METHOD_FFT;
            else if( this->comboBox_ddi_method->currentIndex() == 1 )
                method = SPIRIT_DDI_METHOD_FMM;
            else if( this->comboBox_ddi_method->currentIndex() == 2 )
                method = SPIRIT_DDI_METHOD_CUTOFF;
        }

        int n_periodic_images[3];
        n_periodic_images[0] = this->spinBox_ddi_n_periodic_a->value();
        n_periodic_images[1] = this->spinBox_ddi_n_periodic_b->value();
        n_periodic_images[2] = this->spinBox_ddi_n_periodic_c->value();

        float radius         = this->doubleSpinBox_ddi_radius->value();
        bool pb_zero_padding = this->checkBox_ddi_pb_zero_padding->isChecked();
        Hamiltonian_Set_DDI( state.get(), method, n_periodic_images, radius, pb_zero_padding, idx_image );
    };

    if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int i = 0; i < Chain_Get_NOI( state.get() ); ++i )
        {
            apply( i );
        }
    }
    else if( this->comboBox_Hamiltonian_Ani_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void HamiltonianHeisenbergWidget::set_pairs_from_file()
{
    Log_Send( state.get(), Log_Level_Warning, Log_Sender_UI, "Not yet implemented: set pairs from file" );
}

void HamiltonianHeisenbergWidget::set_pairs_from_text()
{
    Log_Send( state.get(), Log_Level_Warning, Log_Sender_UI, "Not yet implemented: set pairs from text" );
}

// -----------------------------------------------------------------------------------
// --------------------------------- Setup -------------------------------------------
// -----------------------------------------------------------------------------------

void HamiltonianHeisenbergWidget::Setup_Input_Validators()
{
    //      mu_s
    this->lineEdit_muSpin_aniso->setValidator( this->number_validator );
    //      external field
    this->lineEdit_extH_aniso->setValidator( this->number_validator );
    this->lineEdit_extHx_aniso->setValidator( this->number_validator );
    this->lineEdit_extHy_aniso->setValidator( this->number_validator );
    this->lineEdit_extHz_aniso->setValidator( this->number_validator );
    //      anisotropy
    this->lineEdit_ani_aniso->setValidator( this->number_validator );
    this->lineEdit_anix_aniso->setValidator( this->number_validator );
    this->lineEdit_aniy_aniso->setValidator( this->number_validator );
    this->lineEdit_aniz_aniso->setValidator( this->number_validator );
}

void HamiltonianHeisenbergWidget::Setup_Slots()
{
    // Boundary Conditions
    connect(
        this->checkBox_aniso_periodical_a, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );
    connect(
        this->checkBox_aniso_periodical_b, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );
    connect(
        this->checkBox_aniso_periodical_c, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );
    // mu_s
    connect( this->lineEdit_muSpin_aniso, SIGNAL( returnPressed() ), this, SLOT( set_mu_s() ) );
    // External Field
    connect( this->checkBox_extH_aniso, SIGNAL( stateChanged( int ) ), this, SLOT( set_external_field() ) );
    connect( this->lineEdit_extH_aniso, SIGNAL( returnPressed() ), this, SLOT( set_external_field() ) );
    connect( this->lineEdit_extHx_aniso, SIGNAL( returnPressed() ), this, SLOT( set_external_field() ) );
    connect( this->lineEdit_extHy_aniso, SIGNAL( returnPressed() ), this, SLOT( set_external_field() ) );
    connect( this->lineEdit_extHz_aniso, SIGNAL( returnPressed() ), this, SLOT( set_external_field() ) );
    // Anisotropy
    connect( this->checkBox_ani_aniso, SIGNAL( stateChanged( int ) ), this, SLOT( set_anisotropy() ) );
    connect( this->lineEdit_ani_aniso, SIGNAL( returnPressed() ), this, SLOT( set_anisotropy() ) );
    connect( this->lineEdit_anix_aniso, SIGNAL( returnPressed() ), this, SLOT( set_anisotropy() ) );
    connect( this->lineEdit_aniy_aniso, SIGNAL( returnPressed() ), this, SLOT( set_anisotropy() ) );
    connect( this->lineEdit_aniz_aniso, SIGNAL( returnPressed() ), this, SLOT( set_anisotropy() ) );
    // Exchange
    connect( this->checkBox_exchange, SIGNAL( stateChanged( int ) ), this, SLOT( set_exchange() ) );
    connect( this->spinBox_nshells_exchange, SIGNAL( editingFinished() ), this, SLOT( set_nshells_exchange() ) );
    // DMI
    connect( this->checkBox_dmi, SIGNAL( stateChanged( int ) ), this, SLOT( set_dmi() ) );
    connect( this->spinBox_nshells_dmi, SIGNAL( editingFinished() ), this, SLOT( set_nshells_dmi() ) );
    connect( this->comboBox_dmi_chirality, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_dmi() ) );
    // DDI
    connect( this->checkBox_ddi, SIGNAL( stateChanged( int ) ), this, SLOT( set_ddi() ) );
    connect( this->comboBox_ddi_method, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_ddi() ) );
    connect( this->spinBox_ddi_n_periodic_a, SIGNAL( editingFinished() ), this, SLOT( set_ddi() ) );
    connect( this->spinBox_ddi_n_periodic_b, SIGNAL( editingFinished() ), this, SLOT( set_ddi() ) );
    connect( this->spinBox_ddi_n_periodic_c, SIGNAL( editingFinished() ), this, SLOT( set_ddi() ) );
    connect( this->doubleSpinBox_ddi_radius, SIGNAL( editingFinished() ), this, SLOT( set_ddi() ) );
    connect( this->checkBox_ddi_pb_zero_padding, SIGNAL( stateChanged( int ) ), this, SLOT( set_ddi() ) );
    // Pairs
    connect( this->pushButton_pairs_apply, SIGNAL( clicked() ), this, SLOT( set_pairs_from_text() ) );
    connect( this->pushButton_pairs_fromfile, SIGNAL( clicked() ), this, SLOT( set_pairs_from_file() ) );
}