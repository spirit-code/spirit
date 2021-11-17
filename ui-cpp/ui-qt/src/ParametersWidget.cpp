#include "ParametersWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/IO.h>
#include <Spirit/Log.h>
#include <Spirit/Parameters_EMA.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Parameters_MC.h>
#include <Spirit/Parameters_MMF.h>
#include <Spirit/Simulation.h>
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

ParametersWidget::ParametersWidget( std::shared_ptr<State> state )
{
    this->state = state;
    // this->spinWidget = spinWidget;
    // this->settingsWidget = settingsWidget;

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
    QRegularExpression re5( "[\\d.]+(?:e-?\\d+)?" );
    this->number_validator_unsigned_scientific = new QRegularExpressionValidator( re5 );
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
    float d, d2, vd[3];
    int image_type;
    int i1, i2;
    bool b1, b2, b3, b4, b5;

    //      LLG
    // Direct minimization
    b1 = Parameters_LLG_Get_Direct_Minimization( state.get() );
    this->checkBox_llg_direct->setChecked( b1 );
    // Damping
    d = Parameters_LLG_Get_Damping( state.get() );
    this->lineEdit_Damping->setText( QString::number( d ) );
    // Converto to PicoSeconds
    d = Parameters_LLG_Get_Time_Step( state.get() );
    this->lineEdit_dt->setText( QString::number( d ) );
    // Spin polarized current
    Parameters_LLG_Get_STT( state.get(), &b1, &d, vd );
    this->radioButton_stt_gradient->setChecked( b1 );
    this->doubleSpinBox_llg_stt_magnitude->setValue( d );
    this->doubleSpinBox_llg_stt_polarisation_x->setValue( vd[0] );
    this->doubleSpinBox_llg_stt_polarisation_y->setValue( vd[1] );
    this->doubleSpinBox_llg_stt_polarisation_z->setValue( vd[2] );
    if( d > 0.0 )
        this->checkBox_llg_stt->setChecked( true );
    // Temperature
    d = Parameters_LLG_Get_Temperature( state.get() );
    this->doubleSpinBox_llg_temperature->setValue( d );
    if( d > 0.0 )
        this->checkBox_llg_temperature->setChecked( true );
    Parameters_LLG_Get_Temperature_Gradient( state.get(), &d, vd );
    this->lineEdit_llg_temperature_inclination->setText( QString::number( d ) );
    this->lineEdit_llg_temperature_dir_x->setText( QString::number( vd[0] ) );
    this->lineEdit_llg_temperature_dir_y->setText( QString::number( vd[1] ) );
    this->lineEdit_llg_temperature_dir_z->setText( QString::number( vd[2] ) );
    // Convergence
    d = Parameters_LLG_Get_Convergence( state.get() );
    this->spinBox_llg_convergence->setValue( std::log10( d ) );
    // Output
    Parameters_LLG_Get_N_Iterations( state.get(), &i1, &i2 );
    this->lineEdit_llg_n_iterations->setText( QString::number( i1 ) );
    this->lineEdit_llg_log_steps->setText( QString::number( i2 ) );
    auto folder = Parameters_LLG_Get_Output_Folder( state.get() );
    this->lineEdit_llg_output_folder->setText( folder );
    Parameters_LLG_Get_Output_General( state.get(), &b1, &b2, &b3 );
    this->checkBox_llg_output_any->setChecked( b1 );
    this->checkBox_llg_output_initial->setChecked( b2 );
    this->checkBox_llg_output_final->setChecked( b3 );
    Parameters_LLG_Get_Output_Energy( state.get(), &b1, &b2, &b3, &b4, &b5 );
    this->checkBox_llg_output_energy_step->setChecked( b1 );
    this->checkBox_llg_output_energy_archive->setChecked( b2 );
    this->checkBox_llg_output_energy_spin_resolved->setChecked( b3 );
    this->checkBox_llg_output_energy_divide->setChecked( b4 );
    Parameters_LLG_Get_Output_Configuration( state.get(), &b1, &b2, &i1 );
    this->checkBox_llg_output_configuration_step->setChecked( b1 );
    this->checkBox_llg_output_configuration_archive->setChecked( b2 );

    //      MC
    // Parameters
    d = Parameters_MC_Get_Temperature( state.get() );
    this->doubleSpinBox_mc_temperature->setValue( d );
    if( d > 0.0 )
        this->checkBox_mc_temperature->setChecked( true );
    Parameters_MC_Get_Metropolis_Cone( state.get(), &b1, &d2, &b2, &d );
    this->doubleSpinBox_mc_acceptance->setValue( d );
    // Output
    Parameters_MC_Get_N_Iterations( state.get(), &i1, &i2 );
    this->spinBox_mc_n_iterations->setValue( i1 );
    this->spinBox_mc_log_steps->setValue( i2 );
    folder = Parameters_MC_Get_Output_Folder( state.get() );
    this->lineEdit_mc_output_folder->setText( folder );
    Parameters_MC_Get_Output_General( state.get(), &b1, &b2, &b3 );
    this->checkBox_mc_output_any->setChecked( b1 );
    this->checkBox_mc_output_initial->setChecked( b2 );
    this->checkBox_mc_output_final->setChecked( b3 );
    Parameters_MC_Get_Output_Energy( state.get(), &b1, &b2, &b3, &b4, &b5 );
    this->checkBox_mc_output_energy_step->setChecked( b1 );
    this->checkBox_mc_output_energy_archive->setChecked( b2 );
    this->checkBox_mc_output_energy_spin_resolved->setChecked( b3 );
    this->checkBox_mc_output_energy_divide->setChecked( b4 );
    Parameters_MC_Get_Output_Configuration( state.get(), &b1, &b2, &i1 );
    this->checkBox_mc_output_configuration_step->setChecked( b1 );
    this->checkBox_mc_output_configuration_archive->setChecked( b2 );

    //      GNEB
    // Output
    Parameters_GNEB_Get_N_Iterations( state.get(), &i1, &i2 );
    this->lineEdit_gneb_n_iterations->setText( QString::number( i1 ) );
    this->lineEdit_gneb_log_steps->setText( QString::number( i2 ) );
    folder = Parameters_GNEB_Get_Output_Folder( state.get() );
    this->lineEdit_gneb_output_folder->setText( folder );
    Parameters_GNEB_Get_Output_General( state.get(), &b1, &b2, &b3 );
    this->checkBox_gneb_output_any->setChecked( b1 );
    this->checkBox_gneb_output_initial->setChecked( b2 );
    this->checkBox_gneb_output_final->setChecked( b3 );
    Parameters_GNEB_Get_Output_Energies( state.get(), &b1, &b2, &b3, &b4 );
    this->checkBox_gneb_output_energies_step->setChecked( b1 );
    this->checkBox_gneb_output_energies_interpolated->setChecked( b2 );
    this->checkBox_gneb_output_energies_divide->setChecked( b3 );
    Parameters_GNEB_Get_Output_Chain( state.get(), &b1, &i1 );
    this->checkBox_gneb_output_chain_step->setChecked( b1 );

    // Convergence
    d = Parameters_GNEB_Get_Convergence( state.get() );
    this->spinBox_gneb_convergence->setValue( std::log10( d ) );
    // GNEB Spring Constant
    d = Parameters_GNEB_Get_Spring_Constant( state.get() );
    this->lineEdit_gneb_springconstant->setText( QString::number( d ) );
    // Spring force ratio
    d = Parameters_GNEB_Get_Spring_Force_Ratio( state.get() );
    this->lineEdit_gneb_springforceratio->setText( QString::number( d ) );
    // Path shortening constant
    d = Parameters_GNEB_Get_Path_Shortening_Constant( state.get() );
    this->spinBox_gneb_pathshorteningconstant->setValue( std::log10( d ) );

    // Normal/Climbing/Falling image radioButtons
    image_type = Parameters_GNEB_Get_Climbing_Falling( state.get() );
    if( image_type == 0 )
        this->radioButton_Normal->setChecked( true );
    else if( image_type == 1 )
        this->radioButton_ClimbingImage->setChecked( true );
    else if( image_type == 2 )
        this->radioButton_FallingImage->setChecked( true );
    else if( image_type == 3 )
        this->radioButton_Stationary->setChecked( true );

    //      EMA
    // modes to calculate and visualize
    i1 = Parameters_EMA_Get_N_Modes( state.get() );
    this->spinBox_ema_n_modes->setValue( i1 );
    i2 = Parameters_EMA_Get_N_Mode_Follow( state.get() );
    this->spinBox_ema_n_mode_follow->setValue( i2 + 1 );
    this->spinBox_ema_n_mode_follow->setMaximum( i1 );
    d = Parameters_EMA_Get_Frequency( state.get() );
    this->doubleSpinBox_ema_frequency->setValue( d );
    d = Parameters_EMA_Get_Amplitude( state.get() );
    this->doubleSpinBox_ema_amplitude->setValue( d );
    b1 = Parameters_EMA_Get_Snapshot( state.get() );
    this->checkBox_snapshot_mode->setChecked( b1 );

    //       MMF
    // Parameters
    i1 = Parameters_MMF_Get_N_Modes( state.get() );
    this->spinBox_mmf_n_modes->setValue( i1 );
    i2 = Parameters_MMF_Get_N_Mode_Follow( state.get() );
    this->spinBox_mmf_n_mode_follow->setValue( i2 + 1 );
    this->spinBox_mmf_n_mode_follow->setMaximum( i1 );
    // Output
    Parameters_MMF_Get_N_Iterations( state.get(), &i1, &i2 );
    this->lineEdit_mmf_output_n_iterations->setText( QString::number( i1 ) );
    this->lineEdit_mmf_output_log_steps->setText( QString::number( i2 ) );
    auto folder_mmf = Parameters_MMF_Get_Output_Folder( state.get() );
    this->lineEdit_mmf_output_folder->setText( folder_mmf );
    Parameters_MMF_Get_Output_General( state.get(), &b1, &b2, &b3 );
    this->checkBox_mmf_output_any->setChecked( b1 );
    this->checkBox_mmf_output_initial->setChecked( b2 );
    this->checkBox_mmf_output_final->setChecked( b3 );
    Parameters_MMF_Get_Output_Energy( state.get(), &b1, &b2, &b3, &b4, &b5 );
    this->checkBox_mmf_output_energy_step->setChecked( b1 );
    this->checkBox_mmf_output_energy_archive->setChecked( b2 );
    this->checkBox_mmf_output_energy_spin_resolved->setChecked( b3 );
    this->checkBox_mmf_output_energy_divide->setChecked( b4 );
    Parameters_MMF_Get_Output_Configuration( state.get(), &b1, &b2, &i1 );
    this->checkBox_mmf_output_configuration_step->setChecked( b1 );
    this->checkBox_mmf_output_configuration_archive->setChecked( b2 );
}

void ParametersWidget::set_parameters_llg()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        float d, d2, vd[3];
        int i1, i2;
        bool b1, b2, b3, b4;

        // Direct minimization
        b1 = this->checkBox_llg_direct->isChecked();
        Parameters_LLG_Set_Direct_Minimization( this->state.get(), b1, idx_image );

        // Convergence
        d = std::pow( 10, this->spinBox_llg_convergence->value() );
        Parameters_LLG_Set_Convergence( this->state.get(), d, idx_image );

        // Time step [ps]
        // dt = time_step [ps] * 10^-12 * gyromagnetic raio / mu_B  { / (1+damping^2)} <-
        // not implemented
        d = this->lineEdit_dt->text().toFloat();
        Parameters_LLG_Set_Time_Step( this->state.get(), d, idx_image );

        // Damping
        d = this->lineEdit_Damping->text().toFloat();
        Parameters_LLG_Set_Damping( this->state.get(), d, idx_image );

        // Spin polarised current
        b1 = this->radioButton_stt_gradient->isChecked();
        if( this->checkBox_llg_stt->isChecked() )
            d = this->doubleSpinBox_llg_stt_magnitude->value();
        else
            d = 0.0;
        vd[0] = doubleSpinBox_llg_stt_polarisation_x->value();
        vd[1] = doubleSpinBox_llg_stt_polarisation_y->value();
        vd[2] = doubleSpinBox_llg_stt_polarisation_z->value();
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
                Log_Send( state.get(), Log_Level_Warning, Log_Sender_UI, "s_c_vec = {0,0,0} replaced by {0,0,1}" );
                doubleSpinBox_llg_stt_polarisation_x->setValue( 0.0 );
                doubleSpinBox_llg_stt_polarisation_y->setValue( 0.0 );
                doubleSpinBox_llg_stt_polarisation_z->setValue( 1.0 );
            }
            else
            {
                throw( ex );
            }
        }
        Parameters_LLG_Set_STT( state.get(), b1, d, vd, idx_image );

        // Temperature
        if( this->checkBox_llg_temperature->isChecked() )
        {
            d     = this->doubleSpinBox_llg_temperature->value();
            d2    = this->lineEdit_llg_temperature_inclination->text().toFloat();
            vd[0] = this->lineEdit_llg_temperature_dir_x->text().toFloat();
            vd[1] = this->lineEdit_llg_temperature_dir_y->text().toFloat();
            vd[2] = this->lineEdit_llg_temperature_dir_z->text().toFloat();
        }
        else
        {
            d     = 0;
            d2    = 0;
            vd[0] = 0;
            vd[1] = 0;
            vd[2] = 0;
        }
        Parameters_LLG_Set_Temperature( state.get(), d, idx_image );
        Parameters_LLG_Set_Temperature_Gradient( state.get(), d2, vd, idx_image );

        // Output
        i1 = this->lineEdit_llg_n_iterations->text().toInt();
        i2 = this->lineEdit_llg_log_steps->text().toInt();
        Parameters_LLG_Set_N_Iterations( state.get(), i1, i2, idx_image );
        std::string folder = this->lineEdit_llg_output_folder->text().toStdString();
        Parameters_LLG_Set_Output_Folder( state.get(), folder.c_str(), idx_image );
        b1 = this->checkBox_llg_output_any->isChecked();
        b2 = this->checkBox_llg_output_initial->isChecked();
        b3 = this->checkBox_llg_output_final->isChecked();
        Parameters_LLG_Set_Output_General( state.get(), b1, b2, b3, idx_image );
        b1 = this->checkBox_llg_output_energy_step->isChecked();
        b2 = this->checkBox_llg_output_energy_archive->isChecked();
        b3 = this->checkBox_llg_output_energy_spin_resolved->isChecked();
        b4 = this->checkBox_llg_output_energy_divide->isChecked();
        Parameters_LLG_Set_Output_Energy( state.get(), b1, b2, b3, b4, idx_image );
        b1 = this->checkBox_llg_output_configuration_step->isChecked();
        b2 = this->checkBox_llg_output_configuration_archive->isChecked();
        Parameters_LLG_Set_Output_Configuration( state.get(), b1, b2, IO_Fileformat_OVF_text, idx_image );
    };

    if( this->comboBox_LLG_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_LLG_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
    else if( this->comboBox_LLG_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void ParametersWidget::set_parameters_mc()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        float d;
        int i1, i2;
        bool b1, b2, b3, b4;

        // Temperature
        if( this->checkBox_mc_temperature->isChecked() )
            d = this->doubleSpinBox_mc_temperature->value();
        else
            d = 0.0;
        Parameters_MC_Set_Temperature( state.get(), d, idx_image );

        // Acceptance ratio
        d = this->doubleSpinBox_mc_acceptance->value();
        Parameters_MC_Set_Metropolis_Cone( state.get(), true, 40, true, d, idx_image );

        // Output
        i1 = this->spinBox_mc_n_iterations->value();
        i2 = this->spinBox_mc_log_steps->value();
        Parameters_MC_Set_N_Iterations( state.get(), i1, i2, idx_image );
        std::string folder = this->lineEdit_mc_output_folder->text().toStdString();
        Parameters_MC_Set_Output_Folder( state.get(), folder.c_str(), idx_image );
        b1 = this->checkBox_mc_output_any->isChecked();
        b2 = this->checkBox_mc_output_initial->isChecked();
        b3 = this->checkBox_mc_output_final->isChecked();
        Parameters_MC_Set_Output_General( state.get(), b1, b2, b3, idx_image );
        b1 = this->checkBox_mc_output_energy_step->isChecked();
        b2 = this->checkBox_mc_output_energy_archive->isChecked();
        b3 = this->checkBox_mc_output_energy_spin_resolved->isChecked();
        b4 = this->checkBox_mc_output_energy_divide->isChecked();
        Parameters_MC_Set_Output_Energy( state.get(), b1, b2, b3, b4, true, idx_image );
        b1 = this->checkBox_mc_output_configuration_step->isChecked();
        b2 = this->checkBox_mc_output_configuration_archive->isChecked();
        Parameters_MC_Set_Output_Configuration( state.get(), b1, b2, IO_Fileformat_OVF_text, idx_image );
    };

    if( this->comboBox_MC_ApplyTo->currentText() == "Current Image" )
    {
        apply( System_Get_Index( state.get() ) );
    }
    else if( this->comboBox_MC_ApplyTo->currentText() == "Current Image Chain" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
    else if( this->comboBox_MC_ApplyTo->currentText() == "All Images" )
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            apply( img );
        }
    }
}

void ParametersWidget::set_parameters_gneb()
{
    float d;
    int i1, i2;

    // Convergence
    d = std::pow( 10, this->spinBox_gneb_convergence->value() );
    Parameters_GNEB_Set_Convergence( this->state.get(), d, -1 );
    // Spring constant
    d = this->lineEdit_gneb_springconstant->text().toFloat();
    Parameters_GNEB_Set_Spring_Constant( state.get(), d, -1 );
    // Spring force ratio
    d = this->lineEdit_gneb_springforceratio->text().toFloat();
    Parameters_GNEB_Set_Spring_Force_Ratio( state.get(), d );
    // Path shortening force
    if( this->checkBox_gneb_pathshortening->isChecked() )
        d = std::pow( 10, this->spinBox_gneb_pathshorteningconstant->text().toFloat() );
    else
        d = 0;
    Parameters_GNEB_Set_Path_Shortening_Constant( state.get(), d );
    // Climbing/falling image
    int image_type = 0;
    if( this->radioButton_ClimbingImage->isChecked() )
        image_type = 1;
    if( this->radioButton_FallingImage->isChecked() )
        image_type = 2;
    if( this->radioButton_Stationary->isChecked() )
        image_type = 3;
    Parameters_GNEB_Set_Climbing_Falling( state.get(), image_type, -1 );

    // Output
    i1 = this->lineEdit_gneb_n_iterations->text().toInt();
    i2 = this->lineEdit_gneb_log_steps->text().toInt();
    Parameters_GNEB_Set_N_Iterations( state.get(), i1, i2 );
    std::string folder = this->lineEdit_gneb_output_folder->text().toStdString();
    Parameters_GNEB_Set_Output_Folder( state.get(), folder.c_str() );
}

void ParametersWidget::set_gneb_auto_image_type()
{
    Parameters_GNEB_Set_Image_Type_Automatically( state.get() );
    this->Load_Parameters_Contents();
}

void ParametersWidget::set_parameters_mmf()
{
    // Closure to set the parameters of a specific spin system
    auto apply = [this]( int idx_image ) -> void
    {
        int i1, i2;
        bool b1, b2, b3, b4;

        // Parameters
        i1 = this->spinBox_mmf_n_modes->value();
        Parameters_MMF_Set_N_Modes( state.get(), i1, idx_image );
        this->spinBox_mmf_n_mode_follow->setMaximum( i1 );
        i1 = this->spinBox_mmf_n_mode_follow->value();
        Parameters_MMF_Set_N_Mode_Follow( state.get(), i1 - 1, idx_image );

        // Output
        i1 = this->lineEdit_mmf_output_n_iterations->text().toInt();
        i2 = this->lineEdit_mmf_output_log_steps->text().toInt();
        Parameters_MMF_Set_N_Iterations( state.get(), i1, i2, idx_image );
        std::string folder = this->lineEdit_mmf_output_folder->text().toStdString();
        Parameters_MMF_Set_Output_Folder( state.get(), folder.c_str(), idx_image );
        b1 = this->checkBox_mmf_output_any->isChecked();
        b2 = this->checkBox_mmf_output_initial->isChecked();
        b3 = this->checkBox_mmf_output_final->isChecked();
        Parameters_MMF_Set_Output_General( state.get(), b1, b2, b3, idx_image );
        b1 = this->checkBox_mmf_output_energy_step->isChecked();
        b2 = this->checkBox_mmf_output_energy_archive->isChecked();
        b3 = this->checkBox_mmf_output_energy_spin_resolved->isChecked();
        b4 = this->checkBox_mmf_output_energy_divide->isChecked();
        Parameters_MMF_Set_Output_Energy( state.get(), b1, b2, b3, b4, idx_image );
        b1 = this->checkBox_mmf_output_configuration_step->isChecked();
        b2 = this->checkBox_mmf_output_configuration_archive->isChecked();
        Parameters_MMF_Set_Output_Configuration( state.get(), b1, b2, idx_image );
    };

    for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
    {
        apply( img );
    }

    /*if (this->comboBox_MMF_ApplyTo->currentText() == "Current Image")
    {
        apply(System_Get_Index(state.get()), Chain_Get_Index(state.get()));
    }
    else if (this->comboBox_MMF_ApplyTo->currentText() == "Current Image Chain")
    {
        for (int img = 0; img<Chain_Get_NOI(state.get()); ++img)
        {
            apply(img, Chain_Get_Index(state.get()));
        }
    }
    else if (this->comboBox_MMF_ApplyTo->currentText() == "All Images")
    {
        for (int ich = 0; ich<Collection_Get_NOC(state.get()); ++ich)
        {
            for (int img = 0; img<Chain_Get_NOI(state.get(), ich); ++img)
            {
                apply(img, ich);
            }
        }
    }*/
}

void ParametersWidget::set_parameters_ema()
{
    int i1   = this->spinBox_ema_n_modes->value();
    int i2   = this->spinBox_ema_n_mode_follow->value();
    float d1 = this->doubleSpinBox_ema_frequency->value();
    float d2 = this->doubleSpinBox_ema_amplitude->value();
    bool b1  = this->checkBox_snapshot_mode->isChecked();

    Parameters_EMA_Set_N_Modes( state.get(), i1 );
    Parameters_EMA_Set_N_Mode_Follow( state.get(), i2 - 1 );
    Parameters_EMA_Set_Frequency( state.get(), d1 );
    Parameters_EMA_Set_Amplitude( state.get(), d2 );
    Parameters_EMA_Set_Snapshot( state.get(), b1 );

    this->spinBox_ema_n_mode_follow->setMaximum( i1 );
}

void ParametersWidget::save_Spin_Configuration_Eigenmodes()
{
    // std::cerr << "inside save spins" << std::endl;
    auto fileName = QFileDialog::getSaveFileName(
        this, tr( "Save Spin Configuration Eigenmodes" ), "./output", tr( "OOMF Vector Field(*.ovf)" ) );

    int type = IO_Fileformat_OVF_text;

    if( !fileName.isEmpty() )
    {
        QFileInfo fi( fileName );

        // Determine file type from suffix
        auto qs_type = fi.completeSuffix();
        if( qs_type == "ovf" )
            type = IO_Fileformat_OVF_text;

        // Write the file
        auto file = string_q2std( fileName );

        IO_Eigenmodes_Write( this->state.get(), file.c_str(), type );
    }
}

void ParametersWidget::load_Spin_Configuration_Eigenmodes()
{
    auto fileName = QFileDialog::getOpenFileName(
        this, tr( "Load Spin Configuration" ), "./input", tr( "Any (*.txt *.csv *.ovf);;OOMF Vector Field (*.ovf)" ) );

    int type = IO_Fileformat_OVF_text;

    if( !fileName.isEmpty() )
    {
        QFileInfo fi( fileName );
        auto qs_type = fi.suffix();

        if( qs_type == "ovf" )
            type = IO_Fileformat_OVF_text;
        else
            Log_Send(
                state.get(), Log_Level_Error, Log_Sender_UI,
                ( "Invalid file ending (only "
                  "txt, csv and ovf allowed) on file "
                  + string_q2std( fileName ) )
                    .c_str() );

        auto file = string_q2std( fileName );
        IO_Eigenmodes_Read( this->state.get(), file.c_str(), type );

        // n_modes parameter might be change by IO_Eigenmodes_Read so update that first
        this->spinBox_ema_n_modes->setValue( Parameters_EMA_Get_N_Modes( state.get() ) );

        // then pass widgets parameters to the core
        set_parameters_ema();

        // reload parameters to avoid illegal values showing up (eg n_mode_follow)
        updateData();
    }
}

void ParametersWidget::Setup_Parameters_Slots()
{
    //      LLG
    // Direct minimization
    connect( this->checkBox_llg_direct, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    // Temperature
    connect( this->checkBox_llg_temperature, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect( this->doubleSpinBox_llg_temperature, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->lineEdit_llg_temperature_inclination, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_llg_temperature_dir_x, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_llg_temperature_dir_y, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_llg_temperature_dir_z, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    // STT
    connect( this->radioButton_stt_gradient, SIGNAL( clicked() ), this, SLOT( set_parameters_llg() ) );
    connect( this->radioButton_stt_monolayer, SIGNAL( clicked() ), this, SLOT( set_parameters_llg() ) );
    connect( this->checkBox_llg_stt, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect( this->doubleSpinBox_llg_stt_magnitude, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->doubleSpinBox_llg_stt_polarisation_x, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->doubleSpinBox_llg_stt_polarisation_y, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->doubleSpinBox_llg_stt_polarisation_z, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    // Damping
    connect( this->lineEdit_Damping, SIGNAL( returnPressed() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_dt, SIGNAL( returnPressed() ), this, SLOT( set_parameters_llg() ) );
    // Convergence criterion
    connect( this->spinBox_llg_convergence, SIGNAL( editingFinished() ), this, SLOT( set_parameters_llg() ) );
    // Output
    connect( this->lineEdit_llg_n_iterations, SIGNAL( returnPressed() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_llg_log_steps, SIGNAL( returnPressed() ), this, SLOT( set_parameters_llg() ) );
    connect( this->lineEdit_llg_output_folder, SIGNAL( returnPressed() ), this, SLOT( set_parameters_llg() ) );
    connect( this->checkBox_llg_output_any, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect( this->checkBox_llg_output_initial, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect( this->checkBox_llg_output_final, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect( this->checkBox_llg_output_energy_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->checkBox_llg_output_energy_archive, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->checkBox_llg_output_energy_spin_resolved, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_llg() ) );
    connect(
        this->checkBox_llg_output_energy_divide, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_llg() ) );
    connect(
        this->checkBox_llg_output_configuration_step, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_llg() ) );
    connect(
        this->checkBox_llg_output_configuration_archive, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_llg() ) );

    //      MC
    // Paramters
    connect( this->checkBox_mc_temperature, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect( this->doubleSpinBox_mc_acceptance, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mc() ) );
    connect( this->doubleSpinBox_mc_temperature, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mc() ) );
    // Output
    connect( this->spinBox_mc_n_iterations, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mc() ) );
    connect( this->spinBox_mc_log_steps, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mc() ) );
    connect( this->lineEdit_mc_output_folder, SIGNAL( returnPressed() ), this, SLOT( set_parameters_mc() ) );
    connect( this->checkBox_mc_output_any, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect( this->checkBox_mc_output_initial, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect( this->checkBox_mc_output_final, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect( this->checkBox_mc_output_energy_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect(
        this->checkBox_mc_output_energy_archive, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect(
        this->checkBox_mc_output_energy_spin_resolved, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_mc() ) );
    connect( this->checkBox_mc_output_energy_divide, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect(
        this->checkBox_mc_output_configuration_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mc() ) );
    connect(
        this->checkBox_mc_output_configuration_archive, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_mc() ) );

    //      GNEB
    // Spring Constant
    connect( this->lineEdit_gneb_springconstant, SIGNAL( returnPressed() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->lineEdit_gneb_springforceratio, SIGNAL( returnPressed() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->checkBox_gneb_pathshortening, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );
    connect(
        this->spinBox_gneb_pathshorteningconstant, SIGNAL( editingFinished() ), this, SLOT( set_parameters_gneb() ) );
    // Image type
    connect( this->pushButton_auto_image_type, SIGNAL( clicked() ), this, SLOT( set_gneb_auto_image_type() ) );
    connect( this->radioButton_Normal, SIGNAL( clicked() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->radioButton_ClimbingImage, SIGNAL( clicked() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->radioButton_FallingImage, SIGNAL( clicked() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->radioButton_Stationary, SIGNAL( clicked() ), this, SLOT( set_parameters_gneb() ) );
    // Convergence criterion
    connect( this->spinBox_gneb_convergence, SIGNAL( editingFinished() ), this, SLOT( set_parameters_gneb() ) );
    // Output
    connect( this->lineEdit_gneb_n_iterations, SIGNAL( returnPressed() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->lineEdit_gneb_log_steps, SIGNAL( returnPressed() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->lineEdit_gneb_output_folder, SIGNAL( returnPressed() ), this, SLOT( set_parameters_gneb() ) );
    connect( this->checkBox_gneb_output_any, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );
    connect( this->checkBox_gneb_output_initial, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );
    connect( this->checkBox_gneb_output_final, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );
    connect(
        this->checkBox_gneb_output_energies_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );
    connect(
        this->checkBox_gneb_output_energies_divide, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_gneb() ) );
    connect(
        this->checkBox_gneb_output_chain_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_gneb() ) );

    //      EMA
    connect( this->spinBox_ema_n_modes, SIGNAL( editingFinished() ), this, SLOT( set_parameters_ema() ) );
    connect( this->spinBox_ema_n_mode_follow, SIGNAL( editingFinished() ), this, SLOT( set_parameters_ema() ) );
    connect( this->doubleSpinBox_ema_frequency, SIGNAL( editingFinished() ), this, SLOT( set_parameters_ema() ) );
    connect( this->doubleSpinBox_ema_amplitude, SIGNAL( editingFinished() ), this, SLOT( set_parameters_ema() ) );
    connect( this->pushButton_SaveModes, SIGNAL( clicked() ), this, SLOT( save_Spin_Configuration_Eigenmodes() ) );
    connect( this->pushButton_LoadModes, SIGNAL( clicked() ), this, SLOT( load_Spin_Configuration_Eigenmodes() ) );
    connect( this->checkBox_snapshot_mode, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_ema() ) );

    //      MMF
    // Output
    connect( this->lineEdit_mmf_output_n_iterations, SIGNAL( returnPressed() ), this, SLOT( set_parameters_mmf() ) );
    connect( this->lineEdit_mmf_output_log_steps, SIGNAL( returnPressed() ), this, SLOT( set_parameters_mmf() ) );
    connect( this->lineEdit_mmf_output_folder, SIGNAL( returnPressed() ), this, SLOT( set_parameters_mmf() ) );
    connect( this->checkBox_mmf_output_any, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect( this->checkBox_mmf_output_initial, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect( this->checkBox_mmf_output_final, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect( this->checkBox_mmf_output_energy_step, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect(
        this->checkBox_mmf_output_energy_archive, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect(
        this->checkBox_mmf_output_energy_spin_resolved, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_mmf() ) );
    connect(
        this->checkBox_mmf_output_energy_divide, SIGNAL( stateChanged( int ) ), this, SLOT( set_parameters_mmf() ) );
    connect(
        this->checkBox_mmf_output_configuration_step, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_mmf() ) );
    connect(
        this->checkBox_mmf_output_configuration_archive, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_parameters_mmf() ) );
    // Parameters
    connect( this->spinBox_mmf_n_modes, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mmf() ) );
    connect( this->spinBox_mmf_n_mode_follow, SIGNAL( editingFinished() ), this, SLOT( set_parameters_mmf() ) );
    // connect(this->radioButton_mmf_negative_fixed, SIGNAL(clicked()), this,
    // SLOT(set_parameters_mmf())); connect(this->radioButton_mmf_negative_minimum,
    // SIGNAL(clicked()), this, SLOT(set_parameters_mmf()));
    // connect(this->radioButton_mmf_positive_mode_relax, SIGNAL(clicked()), this,
    // SLOT(set_parameters_mmf())); connect(this->radioButton_mmf_positive_mode,
    // SIGNAL(clicked()), this, SLOT(set_parameters_mmf()));
    // connect(this->radioButton_mmf_positive_gradient, SIGNAL(clicked()), this,
    // SLOT(set_parameters_mmf()));
}

void ParametersWidget::Setup_Input_Validators()
{
    //      LLG
    this->lineEdit_Damping->setValidator( this->number_validator_unsigned );
    this->lineEdit_dt->setValidator( this->number_validator_unsigned_scientific );
    this->lineEdit_llg_temperature_inclination->setValidator( this->number_validator );
    this->lineEdit_llg_temperature_dir_x->setValidator( this->number_validator );
    this->lineEdit_llg_temperature_dir_y->setValidator( this->number_validator );
    this->lineEdit_llg_temperature_dir_z->setValidator( this->number_validator );
    //      GNEB
    this->lineEdit_gneb_springconstant->setValidator( this->number_validator_unsigned );
    this->lineEdit_gneb_springforceratio->setValidator( this->number_validator_unsigned );
}
