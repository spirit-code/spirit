#include "ControlWidget.hpp"

#include <QFuture>
#include <QtConcurrent/QtConcurrentRun>
#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/IO.h>
#include <Spirit/Log.h>
#include <Spirit/Parameters_EMA.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

std::string string_q2std( QString qs )
{
    auto bytearray          = qs.toLatin1();
    const char * c_fileName = bytearray.data();
    return std::string( c_fileName );
}

ControlWidget::ControlWidget( std::shared_ptr<State> state, SpinWidget * spinWidget, SettingsWidget * settingsWidget )
{
    this->state          = state;
    this->spinWidget     = spinWidget;
    this->settingsWidget = settingsWidget;

    this->idx_image_last = 0;

    // Create threads
    threads_image = std::vector<std::thread>( Chain_Get_NOI( this->state.get() ) );

    // Setup User Interface
    this->setupUi( this );

    // Buttons
    connect( this->lineEdit_Save_E, SIGNAL( returnPressed() ), this, SLOT( save_EPressed() ) );
    connect( this->pushButton_Save_E, SIGNAL( clicked() ), this, SLOT( save_EPressed() ) );
    connect( this->pushButton_StopAll, SIGNAL( clicked() ), this, SLOT( stop_all() ) );
    connect( this->pushButton_PlayPause, SIGNAL( clicked() ), this, SLOT( play_pause() ) );
    connect( this->pushButton_PreviousImage, SIGNAL( clicked() ), this, SLOT( prev_image() ) );
    connect( this->pushButton_NextImage, SIGNAL( clicked() ), this, SLOT( next_image() ) );
    connect( this->lineEdit_ImageNumber, SIGNAL( returnPressed() ), this, SLOT( jump_to_image() ) );
    connect( this->pushButton_Reset, SIGNAL( clicked() ), this, SLOT( resetPressed() ) );
    connect( this->pushButton_X, SIGNAL( clicked() ), this, SLOT( xPressed() ) );
    connect( this->pushButton_Y, SIGNAL( clicked() ), this, SLOT( yPressed() ) );
    connect( this->pushButton_Z, SIGNAL( clicked() ), this, SLOT( zPressed() ) );
    connect( this->comboBox_Method, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_solver_enabled() ) );
    connect( this->pushButton_PreviousMode, SIGNAL( clicked() ), this, SLOT( prev_mode() ) );
    connect( this->pushButton_NextMode, SIGNAL( clicked() ), this, SLOT( next_mode() ) );
    connect( this->lineEdit_ModeNumber, SIGNAL( returnPressed() ), this, SLOT( jump_to_mode() ) );
    connect( this->pushButton_ApplyMode, SIGNAL( clicked() ), this, SLOT( apply_mode() ) );
    connect( this->pushButton_Calculate, SIGNAL( clicked() ), this, SLOT( calculate() ) );
    connect( &this->watcher, SIGNAL( finished() ), this, SLOT( calculate_enable_widget() ) );

    // Image number
    // We use a regular expression (regex) to filter the input into the lineEdits
    QRegularExpression re( "[\\d]*" );
    QRegularExpressionValidator * number_validator = new QRegularExpressionValidator( re );
    this->lineEdit_ImageNumber->setValidator( number_validator );
    this->lineEdit_ImageNumber->setText( QString::number( 1 ) );
    this->lineEdit_ModeNumber->setValidator( number_validator );
    this->lineEdit_ModeNumber->setText( QString::number( 1 ) );

    ema_buttons_hide();

    // Read persistent settings
    this->readSettings();
}

void ControlWidget::updateData()
{
    // Check for running simulations - update Play/Pause Button
    if( Simulation_Running_On_Chain( state.get() ) || Simulation_Running_On_Image( state.get() ) )
    {
        this->pushButton_PlayPause->setText( "Stop" );
        this->spinWidget->updateData();
    }
    else
    {
        if( this->pushButton_PlayPause->text() == "Stop" )
            this->spinWidget->updateData();
        this->pushButton_PlayPause->setText( "Start" );
    }

    // Update Image number
    int idx_image = System_Get_Index( state.get() );
    // Update Image number
    if( idx_image_last != idx_image )
    {
        this->lineEdit_ImageNumber->setText( QString::number( idx_image + 1 ) );
        this->idx_image_last = idx_image;
    }
    // Update Mode number
    this->lineEdit_ModeNumber->setText( QString::number( Parameters_EMA_Get_N_Mode_Follow( state.get() ) + 1 ) );
    // Update NOI counter
    this->label_NOI->setText( "/ " + QString::number( Chain_Get_NOI( state.get() ) ) );
    // Update NEM counter
    this->label_NumberOfModes->setText( "/ " + QString::number( Parameters_EMA_Get_N_Modes( state.get() ) ) );

    // Update thread arrays
    if( Chain_Get_NOI( state.get() ) > (int)threads_image.size() )
    {
        for( int i = threads_image.size(); i < Chain_Get_NOI( state.get() ); ++i )
            this->threads_image.push_back( std::thread() );
    }
}

void ControlWidget::updateOthers()
{
    // Update the chain's data (primarily for the plot)
    // Chain_Update_Data(state.get());

    // Update Image-dependent Widgets
    this->spinWidget->updateData();
    this->settingsWidget->updateData();
    // this->plotsWidget->updateData();
    // this->debugWidget->updateData();
}

void ControlWidget::cycleMethod()
{
    int idx     = this->comboBox_Method->currentIndex();
    int idx_max = this->comboBox_Method->count();
    this->comboBox_Method->setCurrentIndex( ( idx + 1 ) % idx_max );
}

void ControlWidget::cycleSolver()
{
    int idx     = this->comboBox_Solver->currentIndex();
    int idx_max = this->comboBox_Solver->count();
    this->comboBox_Solver->setCurrentIndex( ( idx + 1 ) % idx_max );
}

std::string ControlWidget::methodName()
{
    return this->comboBox_Method->currentText().toStdString();
}

std::string ControlWidget::solverName()
{
    return this->comboBox_Solver->currentText().toStdString();
}

void ControlWidget::play_pause()
{
    // this->return_focus();

    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: Start/Stop" );

    Chain_Update_Data( this->state.get() );

    auto qs_method = this->comboBox_Method->currentText();
    auto qs_solver = this->comboBox_Solver->currentText();

    this->s_method = string_q2std( qs_method );
    this->s_solver = string_q2std( qs_solver );

    int solver;
    if( s_solver == "VP" )
        solver = Solver_VP;
    else if( s_solver == "SIB" )
        solver = Solver_SIB;
    else if( s_solver == "Depondt" )
        solver = Solver_Depondt;
    else if( s_solver == "Heun" )
        solver = Solver_Heun;
    else if( s_solver == "RK4" )
        solver = Solver_RungeKutta4;
    else if( s_solver == "LBFGS_OSO" )
        solver = Solver_LBFGS_OSO;
    else if( s_solver == "LBFGS_Atlas" )
        solver = Solver_LBFGS_Atlas;
    if( s_solver == "VP_OSO" )
        solver = Solver_VP_OSO;

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
        // New button text
        this->pushButton_PlayPause->setText( "Start" );
    }
    else
    {
        // Not running, so we start it
        if( this->s_method == "LLG" )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_LLG_Start, this->state.get(), solver, -1, -1, false, nullptr, -1, -1 );
        }
        else if( this->s_method == "MC" )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MC_Start, this->state.get(), -1, -1, false, nullptr, -1, -1 );
        }
        else if( this->s_method == "GNEB" )
        {
            if( thread_chain.joinable() )
                thread_chain.join();
            this->thread_chain
                = std::thread( &Simulation_GNEB_Start, this->state.get(), solver, -1, -1, false, nullptr, -1 );
        }
        else if( this->s_method == "MMF" )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MMF_Start, this->state.get(), solver, -1, -1, false, nullptr, -1, -1 );
        }
        else if( this->s_method == "EMA" )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_EMA_Start, this->state.get(), -1, -1, false, nullptr, -1, -1 );
        }
        // New button text
        this->pushButton_PlayPause->setText( "Stop" );
    }
    this->spinWidget->updateData();
}

void ControlWidget::stop_all()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: Stop All" );

    Simulation_Stop_All( state.get() );

    for( unsigned int i = 0; i < threads_image.size(); ++i )
        if( threads_image[i].joinable() )
            threads_image[i].join();
    if( thread_chain.joinable() )
        thread_chain.join();

    this->pushButton_PlayPause->setText( "Start" );
    // this->createStatusBar();
}

void ControlWidget::stop_current()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: Stop All" );

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
    }

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
    }

    this->pushButton_PlayPause->setText( "Start" );
}

void ControlWidget::next_image()
{
    if( System_Get_Index( state.get() ) < Chain_Get_NOI( this->state.get() ) - 1 )
    {
        // Change active image
        Chain_next_Image( this->state.get() );

        // Update
        this->updateData();
        this->updateOthers();
    }
}

void ControlWidget::prev_image()
{
    // this->return_focus();
    if( System_Get_Index( state.get() ) > 0 )
    {
        // Change active image!
        Chain_prev_Image( this->state.get() );

        // Update
        this->updateData();
        this->updateOthers();
    }
}

void ControlWidget::jump_to_image()
{
    // Change active image
    int idx = this->lineEdit_ImageNumber->text().toInt() - 1;
    Chain_Jump_To_Image( this->state.get(), idx );

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::cut_image()
{
    if( Chain_Get_NOI( state.get() ) > 1 )
    {
        this->stop_current();

        Chain_Image_to_Clipboard( state.get() );

        int idx = System_Get_Index( state.get() );
        if( Chain_Delete_Image( state.get(), idx ) )
        {
            // Make the llg_threads vector smaller
            if( this->threads_image[idx].joinable() )
                this->threads_image[idx].join();
            this->threads_image.erase( threads_image.begin() + idx );
        }
    }

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::paste_image( std::string where )
{
    if( where == "current" )
    {
        // Paste a Spin System into current System
        this->stop_current();
        Chain_Replace_Image( state.get() );
    }
    else if( where == "left" )
    {
        int idx = System_Get_Index( state.get() );
        // Insert Image
        Chain_Insert_Image_Before( state.get() );
        // Make the llg_threads vector larger
        this->threads_image.insert( threads_image.begin() + idx, std::thread() );
        // Switch to the inserted image
        Chain_prev_Image( this->state.get() );
    }
    else if( where == "right" )
    {
        int idx = System_Get_Index( state.get() );
        // Insert Image
        Chain_Insert_Image_After( state.get() );
        // Make the llg_threads vector larger
        this->threads_image.insert( threads_image.begin() + idx + 1, std::thread() );
        // Switch to the inserted image
        Chain_next_Image( this->state.get() );
    }

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::delete_image()
{
    if( Chain_Get_NOI( state.get() ) > 1 )
    {
        this->stop_current();

        int idx = System_Get_Index( state.get() );
        if( Chain_Delete_Image( state.get() ) )
        {
            // Make the llg_threads vector smaller
            if( this->threads_image[idx].joinable() )
                this->threads_image[idx].join();
            this->threads_image.erase( threads_image.begin() + idx );
        }

        // Update
        this->updateData();
        this->updateOthers();
    }
}

void ControlWidget::next_mode()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: nextmode" );

    int following_mode = Parameters_EMA_Get_N_Mode_Follow( state.get() );

    // Change mode
    Parameters_EMA_Set_N_Mode_Follow( this->state.get(), following_mode + 1 );

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::prev_mode()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: previousmode" );

    int following_mode = Parameters_EMA_Get_N_Mode_Follow( state.get() );

    // Change mode
    Parameters_EMA_Set_N_Mode_Follow( this->state.get(), following_mode - 1 );

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::jump_to_mode()
{
    // Change active image
    int mode_idx = this->lineEdit_ModeNumber->text().toInt() - 1;

    Parameters_EMA_Set_N_Mode_Follow( this->state.get(), mode_idx - 1 );

    // Update
    this->updateData();
    this->updateOthers();
}

void ControlWidget::calculate()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: calculate" );

    calculate_disable_widget();

    int idx = System_Get_Index( state.get() );
    if( threads_image[idx].joinable() )
        threads_image[System_Get_Index( state.get() )].join();
    if( !Simulation_Running_On_Image( state.get() ) )
        this->threads_image[System_Get_Index( state.get() )]
            = std::thread( &System_Update_Eigenmodes, this->state.get(), -1, -1 );

    QFuture<void> future = QtConcurrent::run( &threads_image[System_Get_Index( state.get() )], &std::thread::join );
    this->watcher.setFuture( future );
}

void ControlWidget::apply_mode()
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Button: apply mode" );

    int following_mode = Parameters_EMA_Get_N_Mode_Follow( state.get() );

    Configuration_Displace_Eigenmode( state.get(), following_mode );

    this->spinWidget->updateData();
}

void ControlWidget::calculate_disable_widget()
{
    this->lineEdit_Save_E->setEnabled( false );
    this->pushButton_Save_E->setEnabled( false );
    this->pushButton_StopAll->setEnabled( false );

    this->pushButton_PlayPause->setEnabled( false );
    this->pushButton_PreviousImage->setEnabled( false );
    this->lineEdit_ImageNumber->setEnabled( false );
    this->pushButton_NextImage->setEnabled( false );
    this->pushButton_ApplyMode->setEnabled( false );

    this->comboBox_Method->setEnabled( false );
    this->comboBox_Solver->setEnabled( false );

    this->pushButton_X->setEnabled( false );
    this->pushButton_Y->setEnabled( false );
    this->pushButton_Z->setEnabled( false );
    this->pushButton_Reset->setEnabled( false );

    this->pushButton_PreviousMode->setEnabled( false );
    this->lineEdit_ModeNumber->setEnabled( false );
    this->pushButton_NextMode->setEnabled( false );
}

void ControlWidget::calculate_enable_widget()
{
    this->lineEdit_Save_E->setEnabled( true );
    this->pushButton_Save_E->setEnabled( true );
    this->pushButton_StopAll->setEnabled( true );

    this->pushButton_PlayPause->setEnabled( true );
    this->pushButton_PreviousImage->setEnabled( true );
    this->lineEdit_ImageNumber->setEnabled( true );
    this->pushButton_NextImage->setEnabled( true );
    this->pushButton_ApplyMode->setEnabled( true );

    this->comboBox_Method->setEnabled( true );
    this->comboBox_Solver->setEnabled( true );

    this->pushButton_X->setEnabled( true );
    this->pushButton_Y->setEnabled( true );
    this->pushButton_Z->setEnabled( true );
    this->pushButton_Reset->setEnabled( true );

    this->pushButton_PreviousMode->setEnabled( true );
    this->lineEdit_ModeNumber->setEnabled( true );
    this->pushButton_NextMode->setEnabled( true );
}

void ControlWidget::resetPressed()
{
    this->spinWidget->setCameraToDefault();
}

void ControlWidget::xPressed()
{
    this->spinWidget->setCameraToX();
}

void ControlWidget::yPressed()
{
    this->spinWidget->setCameraToY();
}

void ControlWidget::zPressed()
{
    this->spinWidget->setCameraToZ();
}

void ControlWidget::save_Energies()
{
    // this->return_focus();
    auto fileName = QFileDialog::getSaveFileName( this, tr( "Save Energies" ), "./output", tr( "Text (*.txt)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Chain_Write_Energies( this->state.get(), file.c_str() );
    }
}

void ControlWidget::set_solver_enabled()
{
    if( this->comboBox_Method->currentText() == "MC" )
    {
        this->comboBox_Solver->setEnabled( false );
    }
    else if( this->comboBox_Method->currentText() == "EMA" )
    {
        this->comboBox_Solver->setEnabled( false );
        ema_buttons_show();
    }
    else
    {
        this->comboBox_Solver->setEnabled( true );
        ema_buttons_hide();
    }
}

void ControlWidget::ema_buttons_hide()
{
    this->pushButton_Calculate->setVisible( false );
    this->pushButton_ApplyMode->setVisible( false );
    this->pushButton_NextMode->setVisible( false );
    this->pushButton_PreviousMode->setVisible( false );
    this->lineEdit_ModeNumber->setVisible( false );
    this->label_NumberOfModes->setVisible( false );
}

void ControlWidget::ema_buttons_show()
{
    this->pushButton_Calculate->setVisible( true );
    this->pushButton_ApplyMode->setVisible( true );
    this->pushButton_NextMode->setVisible( true );
    this->pushButton_PreviousMode->setVisible( true );
    this->lineEdit_ModeNumber->setVisible( true );
    this->label_NumberOfModes->setVisible( true );
}

void ControlWidget::save_EPressed()
{
    std::string fullName             = "output/";
    std::string fullNameSpins        = "output/";
    std::string fullNameInterpolated = "output/";

    // Get file info
    auto qFileName = lineEdit_Save_E->text();
    QFileInfo fileInfo( qFileName );

    // Construct the file names
    std::string fileName = string_q2std( fileInfo.baseName() ) + "." + string_q2std( fileInfo.completeSuffix() );
    std::string fileNameSpins
        = string_q2std( fileInfo.baseName() ) + "_Spins." + string_q2std( fileInfo.completeSuffix() );
    std::string fileNameInterpolated
        = string_q2std( fileInfo.baseName() ) + "_Interpolated." + string_q2std( fileInfo.completeSuffix() );

    // File names including path
    fullName.append( fileName );
    fullNameSpins.append( fileNameSpins );
    fullNameInterpolated.append( fileNameInterpolated );

    // Save Energies and Energies_Spins
    IO_Image_Write_Energy_per_Spin( this->state.get(), fullNameSpins.c_str(), IO_Fileformat_OVF_text );
    IO_Chain_Write_Energies( this->state.get(), fullName.c_str() );
    IO_Chain_Write_Energies_Interpolated( this->state.get(), fullNameInterpolated.c_str() );

    // Update File name in LineEdit if it fits the schema
    size_t found = fileName.find( "Energies" );
    if( found != std::string::npos )
    {
        int a = std::stoi( fileName.substr( found + 9, 3 ) ) + 1;
        char newName[20];
        snprintf( newName, 20, "Energies_%03i.txt", a );
        lineEdit_Save_E->setText( newName );
    }
}

void ControlWidget::readSettings()
{
    QSettings settings( "Spirit Code", "Spirit" );

    // Method and Solver
    settings.beginGroup( "ControlWidget" );
    this->comboBox_Method->setCurrentIndex( settings.value( "Method" ).toInt() );
    this->comboBox_Solver->setCurrentIndex( settings.value( "Solver" ).toInt() );
    settings.endGroup();
}

void ControlWidget::writeSettings()
{
    QSettings settings( "Spirit Code", "Spirit" );

    // Method and Solver
    settings.beginGroup( "ControlWidget" );
    settings.setValue( "Method", this->comboBox_Method->currentIndex() );
    settings.setValue( "Solver", this->comboBox_Solver->currentIndex() );
    settings.endGroup();
}

void ControlWidget::closeEvent( QCloseEvent * event )
{
    writeSettings();
    event->accept();
}