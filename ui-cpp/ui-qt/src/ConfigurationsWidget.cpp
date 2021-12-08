#include "ConfigurationsWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/System.h>
#include <Spirit/Transitions.h>

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

ConfigurationsWidget::ConfigurationsWidget( std::shared_ptr<State> state, SpinWidget * spinWidget )
{
    this->state      = state;
    this->spinWidget = spinWidget;

    // Setup User Interface
    this->setupUi( this );
    this->groupBox_globule->setVisible( false );
    this->groupBox_hopfion->setVisible( true );

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

    // Defaults
    this->last_configuration = "";

    // Setup Transitions Tab
    this->lineEdit_Transition_Homogeneous_Last->setText( QString::number( Chain_Get_NOI( this->state.get() ) ) );

    // Load variables from SpinWidget and State
    this->updateData();

    // Connect signals and slots
    this->Setup_Configurations_Slots();
    this->Setup_Transitions_Slots();
}

void ConfigurationsWidget::updateData() {}

void ConfigurationsWidget::print_Energies_to_console()
{
    System_Update_Data( state.get() );
    System_Print_Energy_Array( state.get() );
}

// -----------------------------------------------------------------------------------
// --------------------- Configurations ----------------------------------------------
// -----------------------------------------------------------------------------------
void ConfigurationsWidget::lastConfiguration()
{
    if( last_configuration != "" )
    {
        Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Inserting last used configuration" );
        if( this->last_configuration == "hopfion" )
            this->create_Hopfion();
        else if( this->last_configuration == "skyrmion" )
            this->create_Skyrmion();
        else if( this->last_configuration == "spinspiral" )
            this->create_SpinSpiral();
        else if( this->last_configuration == "domain" )
            this->domainPressed();
    }
}

void ConfigurationsWidget::randomPressed()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Random" );
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    Configuration_Random( this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::addNoisePressed()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Add Noise" );
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::minusZ()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Minus Z" );
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    Configuration_MinusZ( this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::plusZ()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Plus Z" );
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    Configuration_PlusZ( this->state.get(), pos.data(), border_rect.data(), border_cyl, border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::create_Hopfion()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Create Hopfion" );
    this->last_configuration = "hopfion";
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    float r   = lineEdit_hopfion_radius->text().toFloat();
    int order = lineEdit_hopfion_order->text().toInt();

    float normal[3];
    normal[0] = lineEdit_hopfion_normal_x->text().toFloat();
    normal[1] = lineEdit_hopfion_normal_y->text().toFloat();
    normal[2] = lineEdit_hopfion_normal_z->text().toFloat();

    Configuration_Hopfion(
        this->state.get(), r, order, pos.data(), border_rect.data(), border_cyl, border_sph, inverted, normal );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::create_Skyrmion()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Create Skyrmion" );
    this->last_configuration = "skyrmion";
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    float rad      = lineEdit_skyrmion_radius->text().toFloat();
    float order    = lineEdit_skyrmion_order->text().toFloat();
    float phase    = lineEdit_skyrmion_phase->text().toFloat();
    bool upDown    = checkBox_skyrmion_UpDown->isChecked();
    bool achiral   = checkBox_skyrmion_achiral->isChecked();
    bool rl        = checkBox_skyrmion_RL->isChecked();
    bool dw        = checkBox_DW->isChecked();
    float dw_width = lineEdit_DW->text().toFloat();
    // bool experimental = checkBox_sky_experimental->isChecked();

    if( !dw )
        Configuration_Skyrmion(
            this->state.get(), rad, order, phase, upDown, achiral, rl, pos.data(), border_rect.data(), border_cyl,
            border_sph, inverted );
    else
        Configuration_DW_Skyrmion(
            this->state.get(), rad, dw_width, order, phase, upDown, achiral, rl, pos.data(), border_rect.data(),
            border_cyl, border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::create_SpinSpiral()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button createSpinSpiral" );
    this->last_configuration = "spinspiral";
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    float angle   = lineEdit_SS_angle->text().toFloat();
    float axis[3] = { lineEdit_SS_axis_x->text().toFloat(), lineEdit_SS_axis_y->text().toFloat(),
                      lineEdit_SS_axis_z->text().toFloat() };
    float qmag    = this->doubleSpinBox_spiral_q->value();
    float qvec[3] = { lineEdit_SS_dir_x->text().toFloat(), lineEdit_SS_dir_y->text().toFloat(),
                      lineEdit_SS_dir_z->text().toFloat() };

    // Normalize qvec
    float absq = std::sqrt( qvec[0] * qvec[0] + qvec[1] * qvec[1] + qvec[2] * qvec[2] );
    if( absq == 0 )
    {
        qvec[0] = 0;
        qvec[1] = 0;
        qvec[2] = 1;
        lineEdit_SS_dir_x->setText( QString::number( qvec[0] ) );
        lineEdit_SS_dir_y->setText( QString::number( qvec[1] ) );
        lineEdit_SS_dir_z->setText( QString::number( qvec[2] ) );
    }

    // Scale
    for( int dim = 0; dim < 3; ++dim )
        qvec[dim] *= qmag;

    const char * direction_type;
    if( comboBox_SS->currentText() == "Real Lattice" )
        direction_type = "Real Lattice";
    else if( comboBox_SS->currentText() == "Reciprocal Lattice" )
        direction_type = "Reciprocal Lattice";
    else if( comboBox_SS->currentText() == "Real Space" )
        direction_type = "Real Space";

    // Create Spin Spiral
    if( this->checkBox_SS_q2->isChecked() )
    {
        float qvec2[3] = { (float)doubleSpinBox_SS_q2_dir_x->value(), (float)doubleSpinBox_SS_q2_dir_y->value(),
                           (float)doubleSpinBox_SS_q2_dir_z->value() };

        float qmag2 = this->doubleSpinBox_spiral_q2->value();

        // Normalize qvec2
        float absq2 = std::sqrt( qvec2[0] * qvec2[0] + qvec2[1] * qvec2[1] + qvec2[2] * qvec2[2] );
        if( absq == 0 )
        {
            qvec2[0] = 0;
            qvec2[1] = 0;
            qvec2[2] = 1;
            doubleSpinBox_SS_q2_dir_x->setValue( qvec2[0] );
            doubleSpinBox_SS_q2_dir_y->setValue( qvec2[1] );
            doubleSpinBox_SS_q2_dir_z->setValue( qvec2[2] );
        }
        // Scale
        for( int dim = 0; dim < 3; ++dim )
            qvec2[dim] *= qmag2;
        Configuration_SpinSpiral_2q(
            this->state.get(), direction_type, qvec, qvec2, axis, angle, pos.data(), border_rect.data(), border_cyl,
            border_sph, inverted );
    }
    else
        Configuration_SpinSpiral(
            this->state.get(), direction_type, qvec, axis, angle, pos.data(), border_rect.data(), border_cyl,
            border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::domainPressed()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "button Domain" );
    this->last_configuration = "domain";
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Create configuration
    float dir[3] = { lineEdit_domain_dir_x->text().toFloat(), lineEdit_domain_dir_y->text().toFloat(),
                     lineEdit_domain_dir_z->text().toFloat() };
    Configuration_Domain( this->state.get(), dir, pos.data(), border_rect.data(), border_cyl, border_sph, inverted );

    // Optionally add noise
    this->configurationAddNoise();
    print_Energies_to_console();
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::set_atom_type_pressed()
{
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Set
    int atom_type = lineEdit_atom_type->text().toInt();
    Configuration_Set_Atom_Type(
        this->state.get(), atom_type, pos.data(), border_rect.data(), border_cyl, border_sph, inverted );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::set_pinned_pressed()
{
    // Get settings
    auto pos         = get_position();
    auto border_rect = get_border_rectangular();
    float border_cyl = get_border_cylindrical();
    float border_sph = get_border_spherical();
    bool inverted    = get_inverted();
    // Set
    Configuration_Set_Pinned(
        this->state.get(), this->checkBox_pinned->isChecked(), pos.data(), border_rect.data(), border_cyl, border_sph,
        inverted );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::configurationAddNoise()
{
    // Add Noise
    if( this->checkBox_Configuration_Noise->isChecked() )
    {
        // Get settings
        auto pos         = get_position();
        auto border_rect = get_border_rectangular();
        float border_cyl = get_border_cylindrical();
        float border_sph = get_border_spherical();
        bool inverted    = get_inverted();
        // Create configuration
        float temperature = lineEdit_Configuration_Noise->text().toFloat();
        Configuration_Add_Noise_Temperature(
            this->state.get(), temperature, pos.data(), border_rect.data(), border_cyl, border_sph, inverted );

        Chain_Update_Data( this->state.get() );
        this->spinWidget->updateData();
    }
}

// -----------------------------------------------------------------------------------
// --------------------- Transitions -------------------------------------------------
// -----------------------------------------------------------------------------------

void ConfigurationsWidget::homogeneousTransitionPressed()
{
    int idx_1 = this->lineEdit_Transition_Homogeneous_First->text().toInt() - 1;
    int idx_2 = this->lineEdit_Transition_Homogeneous_Last->text().toInt() - 1;

    int noi = Chain_Get_NOI( this->state.get() );

    // Check the validity of the indices
    if( idx_1 < 0 || idx_1 >= noi )
    {
        Log_Send(
            state.get(), Log_Level_Error, Log_Sender_UI,
            "First index for homogeneous transition is invalid! setting to 1..." );
        this->lineEdit_Transition_Homogeneous_First->setText( QString::number( 1 ) );
        return;
    }
    if( idx_2 < 0 || idx_2 >= noi )
    {
        Log_Send(
            state.get(), Log_Level_Error, Log_Sender_UI,
            "Second index for homogeneous transition is invalid! setting to NOI..." );
        this->lineEdit_Transition_Homogeneous_Last->setText( QString::number( noi ) );
        return;
    }
    if( idx_1 == idx_2 )
    {
        Log_Send(
            state.get(), Log_Level_Error, Log_Sender_UI, "Indices are equal in homogeneous transition! Aborting..." );
        return;
    }
    if( idx_2 < idx_1 )
    {
        Log_Send(
            state.get(), Log_Level_Error, Log_Sender_UI,
            "Index 2 is smaller than index 1 in homogeneous transition! Aborting..." );
        return;
    }

    // Do the transition
    Transition_Homogeneous( this->state.get(), idx_1, idx_2 );

    // Add Noise
    if( this->checkBox_Transition_Noise->isChecked() )
    {
        float temperature = lineEdit_Transition_Noise->text().toFloat();
        Transition_Add_Noise_Temperature( this->state.get(), temperature, idx_1, idx_2 );
    }

    // Update
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::homogeneousTransitionFirstLastPressed()
{
    int noi = Chain_Get_NOI( this->state.get() );

    int idx_1 = 0;
    int idx_2 = noi - 1;

    // Do the transition
    Transition_Homogeneous( this->state.get(), idx_1, idx_2 );

    // Add Noise
    if( this->checkBox_Transition_Noise->isChecked() )
    {
        float temperature = lineEdit_Transition_Noise->text().toFloat();
        Transition_Add_Noise_Temperature( this->state.get(), temperature, idx_1, idx_2 );
    }

    // Update
    Chain_Update_Data( this->state.get() );
    this->spinWidget->updateData();
}

void ConfigurationsWidget::homogeneousTransitionInterpolatePressed()
{
    int noi           = Chain_Get_NOI( this->state.get() );
    int n_interpolate = lineEdit_Transition_Homogeneous_Interpolate->text().toFloat();

    if( n_interpolate == 0 || noi < 2 )
    {
        return;
    }
    Transition_Homogeneous_Insert_Interpolated( state.get(), n_interpolate );
}

// -----------------------------------------------------------------------------------
// -------------- Helpers for fetching Configurations Settings -----------------------
// -----------------------------------------------------------------------------------
std::array<float, 3> ConfigurationsWidget::get_position()
{
    return std::array<float, 3>{ lineEdit_pos_x->text().toFloat(), lineEdit_pos_y->text().toFloat(),
                                 lineEdit_pos_z->text().toFloat() };
}

std::array<float, 3> ConfigurationsWidget::get_border_rectangular()
{
    std::array<float, 3> ret{ -1, -1, -1 };
    if( checkBox_border_rectangular_x->isChecked() )
        ret[0] = lineEdit_border_x->text().toFloat();
    if( checkBox_border_rectangular_y->isChecked() )
        ret[1] = lineEdit_border_y->text().toFloat();
    if( checkBox_border_rectangular_z->isChecked() )
        ret[2] = lineEdit_border_z->text().toFloat();
    return ret;
}

float ConfigurationsWidget::get_border_cylindrical()
{
    if( checkBox_border_cylindrical->isChecked() )
    {
        return lineEdit_border_cylindrical->text().toFloat();
    }
    else
    {
        return -1;
    }
}

float ConfigurationsWidget::get_border_spherical()
{
    if( checkBox_border_spherical->isChecked() )
    {
        return lineEdit_border_spherical->text().toFloat();
    }
    else
    {
        return -1;
    }
}

float ConfigurationsWidget::get_inverted()
{
    return checkBox_inverted->isChecked();
}

// -----------------------------------------------------------------------------------
// --------------------------------- Setup -------------------------------------------
// -----------------------------------------------------------------------------------

void ConfigurationsWidget::Setup_Input_Validators()
{
    // Configurations
    //      Settings
    this->lineEdit_Configuration_Noise->setValidator( this->number_validator_unsigned );
    this->lineEdit_pos_x->setValidator( this->number_validator );
    this->lineEdit_pos_y->setValidator( this->number_validator );
    this->lineEdit_pos_z->setValidator( this->number_validator );
    this->lineEdit_border_x->setValidator( this->number_validator_unsigned );
    this->lineEdit_border_y->setValidator( this->number_validator_unsigned );
    this->lineEdit_border_z->setValidator( this->number_validator_unsigned );
    this->lineEdit_border_cylindrical->setValidator( this->number_validator_unsigned );
    this->lineEdit_border_spherical->setValidator( this->number_validator_unsigned );
    //      Hopfion
    this->lineEdit_hopfion_radius->setValidator( this->number_validator );
    this->lineEdit_hopfion_order->setValidator( this->number_validator_int_unsigned );
    //      Skyrmion
    this->lineEdit_skyrmion_order->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_skyrmion_phase->setValidator( this->number_validator );
    this->lineEdit_skyrmion_radius->setValidator( this->number_validator );
    this->lineEdit_DW->setValidator( this->number_validator );

    //      Spin Spiral
    this->lineEdit_SS_dir_x->setValidator( this->number_validator );
    this->lineEdit_SS_dir_y->setValidator( this->number_validator );
    this->lineEdit_SS_dir_z->setValidator( this->number_validator );
    this->lineEdit_SS_axis_x->setValidator( this->number_validator );
    this->lineEdit_SS_axis_y->setValidator( this->number_validator );
    this->lineEdit_SS_axis_z->setValidator( this->number_validator );
    this->lineEdit_SS_angle->setValidator( this->number_validator );
    //      Domain
    this->lineEdit_domain_dir_x->setValidator( this->number_validator );
    this->lineEdit_domain_dir_y->setValidator( this->number_validator );
    this->lineEdit_domain_dir_z->setValidator( this->number_validator );

    //      Atom types
    this->lineEdit_atom_type->setValidator( this->number_validator_int );

    // Transitions
    this->lineEdit_Transition_Noise->setValidator( this->number_validator_unsigned );
    this->lineEdit_Transition_Homogeneous_First->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_Transition_Homogeneous_Last->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_Transition_Homogeneous_Interpolate->setValidator( this->number_validator_int_unsigned );
}

void ConfigurationsWidget::DW_Width_CheckBox_State_Changed()
{
    if( checkBox_DW->isChecked() )
    {
        lineEdit_DW->setEnabled( true );
    }
    else
    {
        lineEdit_DW->setEnabled( false );
    }
}

void ConfigurationsWidget::Setup_Configurations_Slots()
{
    // Random
    connect( this->pushButton_Random, SIGNAL( clicked() ), this, SLOT( randomPressed() ) );
    // Add Noise
    connect( this->pushButton_AddNoise, SIGNAL( clicked() ), this, SLOT( addNoisePressed() ) );
    // Domain
    connect( this->pushButton_domain, SIGNAL( clicked() ), this, SLOT( domainPressed() ) );
    // Homogeneous
    connect( this->pushButton_plusZ, SIGNAL( clicked() ), this, SLOT( plusZ() ) );
    connect( this->pushButton_minusZ, SIGNAL( clicked() ), this, SLOT( minusZ() ) );
    // Hopfion
    connect( this->pushButton_hopfion, SIGNAL( clicked() ), this, SLOT( create_Hopfion() ) );
    // Skyrmion
    connect( this->pushButton_skyrmion, SIGNAL( clicked() ), this, SLOT( create_Skyrmion() ) );
    // Spin Spiral
    connect( this->pushButton_SS, SIGNAL( clicked() ), this, SLOT( create_SpinSpiral() ) );

    // Domain  LineEdits
    connect( this->lineEdit_domain_dir_x, SIGNAL( returnPressed() ), this, SLOT( domainPressed() ) );
    connect( this->lineEdit_domain_dir_y, SIGNAL( returnPressed() ), this, SLOT( domainPressed() ) );
    connect( this->lineEdit_domain_dir_z, SIGNAL( returnPressed() ), this, SLOT( domainPressed() ) );

    // Hopfion LineEdits
    connect( this->lineEdit_hopfion_radius, SIGNAL( returnPressed() ), this, SLOT( create_Hopfion() ) );
    connect( this->lineEdit_hopfion_order, SIGNAL( returnPressed() ), this, SLOT( create_Hopfion() ) );

    // Skyrmion LineEdits
    connect( this->lineEdit_skyrmion_order, SIGNAL( returnPressed() ), this, SLOT( create_Skyrmion() ) );
    connect( this->lineEdit_skyrmion_phase, SIGNAL( returnPressed() ), this, SLOT( create_Skyrmion() ) );
    connect( this->lineEdit_skyrmion_radius, SIGNAL( returnPressed() ), this, SLOT( create_Skyrmion() ) );
    connect( this->lineEdit_DW, SIGNAL( returnPressed() ), this, SLOT( create_Skyrmion() ) );
    connect( this->checkBox_DW, SIGNAL( stateChanged( int ) ), this, SLOT( DW_Width_CheckBox_State_Changed() ) );

    // SpinSpiral
    connect( this->doubleSpinBox_spiral_q, SIGNAL( editingFinished() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_dir_x, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_dir_y, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_dir_z, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_axis_x, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_axis_y, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_axis_z, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->lineEdit_SS_angle, SIGNAL( returnPressed() ), this, SLOT( create_SpinSpiral() ) );
    // q2
    connect( this->checkBox_SS_q2, SIGNAL( stateChanged( int ) ), this, SLOT( create_SpinSpiral() ) );
    connect( this->doubleSpinBox_spiral_q2, SIGNAL( editingFinished() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->doubleSpinBox_SS_q2_dir_x, SIGNAL( editingFinished() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->doubleSpinBox_SS_q2_dir_y, SIGNAL( editingFinished() ), this, SLOT( create_SpinSpiral() ) );
    connect( this->doubleSpinBox_SS_q2_dir_z, SIGNAL( editingFinished() ), this, SLOT( create_SpinSpiral() ) );

    // Pinning and atom types
    connect( this->pushButton_set_atom_type, SIGNAL( clicked() ), this, SLOT( set_atom_type_pressed() ) );
    connect( this->pushButton_set_pinned, SIGNAL( clicked() ), this, SLOT( set_pinned_pressed() ) );
    connect(
        this->lineEdit_atom_type, &QLineEdit::returnPressed, this,
        [this] { this->spinWidget->setPasteAtomType( this->lineEdit_atom_type->text().toInt() ); } );
}

void ConfigurationsWidget::Setup_Transitions_Slots()
{
    // Homogeneous Transition
    connect(
        this->pushButton_Transition_Homogeneous, SIGNAL( clicked() ), this, SLOT( homogeneousTransitionPressed() ) );
    connect(
        this->pushButton_Transition_Homogeneous_First_Last, SIGNAL( clicked() ), this,
        SLOT( homogeneousTransitionFirstLastPressed() ) );
    connect(
        this->pushButton_Transition_Homogeneous_Interpolate, SIGNAL( clicked() ), this,
        SLOT( homogeneousTransitionInterpolatePressed() ) );
}