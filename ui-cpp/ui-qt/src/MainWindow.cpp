#include "MainWindow.hpp"
#include "PlotWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/IO.h>
#include <Spirit/Log.h>
#include <Spirit/Quantities.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#include <algorithm>
#include <locale>

MainWindow::MainWindow( std::shared_ptr<State> state )
{
    // Fix the locale, which QApplication changes on creation
    auto old = std::locale::global( std::locale::classic() );
    std::locale::global( old );

    // State
    this->state = state;
    // Widgets
    this->spinWidget     = new SpinWidget( this->state );
    this->settingsWidget = new SettingsWidget( this->state, this->spinWidget );
    this->plotsWidget    = new PlotsWidget( this->state );
    this->debugWidget    = new DebugWidget( this->state );
    this->controlWidget  = new ControlWidget( this->state, this->spinWidget, this->settingsWidget );
    this->infoWidget     = new InfoWidget( this->state, this->spinWidget, this->controlWidget );

    // this->setFocus(Qt::StrongFocus);
    this->setFocusPolicy( Qt::StrongFocus );

// Fix text size
#ifdef Q_OS_MAC
    this->setStyleSheet( "QWidget{font-size:10pt}" );
#else
    this->setStyleSheet( "QWidget{font-size:8pt}" );
#endif

    // Setup User Interface
    this->setupUi( this );

    // DockWidgets: tabify for Plots and Debug
    this->tabifyDockWidget( this->dockWidget_Plots, this->dockWidget_Debug );
    this->dockWidget_Plots->raise();
    this->dockWidget_Debug->hide();
    // DockWidgets: assign widgets
    this->dockWidget_Settings->setWidget( this->settingsWidget );
    this->dockWidget_Plots->setWidget( this->plotsWidget );
    this->dockWidget_Debug->setWidget( this->debugWidget );

    // Add Widgets to UIs grids
    this->gridLayout->addWidget( this->spinWidget, 0, 0, 1, 1 );
    this->gridLayout->addWidget( this->infoWidget, 0, 0, 1, 1 );
    this->gridLayout_2->addWidget( this->controlWidget, 0, 0, 1, 1 );

    // Read Window settings of last session
    this->view_spins_only    = false;
    this->view_fullscreen    = false;
    this->m_spinWidgetActive = true;
    this->m_InfoWidgetActive = true;
    Ui::MainWindow::statusBar->hide();
    readSettings();

    // File Menu
    connect( this->actionLoad_Configuration, SIGNAL( triggered() ), this, SLOT( load_Configuration() ) );
    connect( this->actionSave_Cfg_File, SIGNAL( triggered() ), this, SLOT( save_Configuration() ) );
    connect( this->actionLoad_Spin_Configuration, SIGNAL( triggered() ), this, SLOT( load_Spin_Configuration() ) );
    connect(
        this->actionLoad_Spin_Configuration_Eigenmodes, SIGNAL( triggered() ), this,
        SLOT( load_Spin_Configuration_Eigenmodes() ) );
    connect(
        this->actionLoad_Spin_Configuration_Chain, SIGNAL( triggered() ), this,
        SLOT( load_Spin_Configuration_Chain() ) );
    connect( this->actionSave_Energy_per_Spin, SIGNAL( triggered() ), this, SLOT( save_System_Energy_Spins() ) );
    connect( this->actionSave_Energies, SIGNAL( triggered() ), this, SLOT( save_Chain_Energies() ) );
    connect(
        this->actionSave_Energies_Interpolated, SIGNAL( triggered() ), this,
        SLOT( save_Chain_Energies_Interpolated() ) );
    connect( this->action_Save_Spin_Configuration, SIGNAL( triggered() ), SLOT( save_Spin_Configuration() ) );
    connect(
        this->actionSave_Spin_Configuration_Eigenmodes, SIGNAL( triggered() ),
        SLOT( save_Spin_Configuration_Eigenmodes() ) );
    connect(
        this->actionSave_Spin_Configuration_Chain, SIGNAL( triggered() ), this,
        SLOT( save_Spin_Configuration_Chain() ) );
    connect( this->actionTake_Screenshot, SIGNAL( triggered() ), this, SLOT( takeScreenshot() ) );
    connect( this->actionTake_Screenshot_Chain, SIGNAL( triggered() ), this, SLOT( take_Screenshot_Chain() ) );

    // Edit Menu
    connect( this->actionCut_Configuration, SIGNAL( triggered() ), this, SLOT( edit_cut() ) );
    connect( this->actionCopy_Configuration, SIGNAL( triggered() ), this, SLOT( edit_copy() ) );
    connect( this->actionPaste_Configuration, SIGNAL( triggered() ), this, SLOT( edit_paste() ) );
    connect( this->actionInsert_Left, SIGNAL( triggered() ), this, SLOT( edit_insert_left() ) );
    connect( this->actionInsert_Right, SIGNAL( triggered() ), this, SLOT( edit_insert_right() ) );
    connect( this->actionDelete_Configuration, SIGNAL( triggered() ), this, SLOT( edit_delete() ) );

    // Control Menu
    connect( this->actionPlay_Pause_Simulation, SIGNAL( triggered() ), this, SLOT( control_playpause() ) );
    connect( this->actionRandomize_Spins, SIGNAL( triggered() ), this, SLOT( control_random() ) );
    connect( this->actionCycle_Method, SIGNAL( triggered() ), this, SLOT( control_cycle_method() ) );
    connect( this->actionCycle_Solver, SIGNAL( triggered() ), this, SLOT( control_cycle_solver() ) );
    connect(
        this->actionToggle_Dragging_mode, &QAction::triggered, this,
        [this] { view_togglePasteMode( SpinWidget::InteractionMode::DRAG ); } );
    connect(
        this->actionToggle_Defect_mode, &QAction::triggered, this,
        [this] { view_togglePasteMode( SpinWidget::InteractionMode::DEFECT ); } );
    connect(
        this->actionToggle_Pinning_mode, &QAction::triggered, this,
        [this] { view_togglePasteMode( SpinWidget::InteractionMode::PIN ); } );

    // View Menu
    connect( this->actionShow_Settings, SIGNAL( triggered() ), this, SLOT( view_toggleSettings() ) );
    connect( this->actionShow_Plots, SIGNAL( triggered() ), this, SLOT( view_togglePlots() ) );
    connect( this->actionShow_Debug, SIGNAL( triggered() ), this, SLOT( view_toggleDebug() ) );
    connect( this->actionToggle_Geometry, SIGNAL( triggered() ), this, SLOT( view_toggleGeometry() ) );
    connect( this->actionToggle_camera_projection, SIGNAL( triggered() ), this, SLOT( view_cycle_camera() ) );
    connect( this->actionRegular_mode, SIGNAL( triggered() ), this, SLOT( view_regular_mode() ) );
    connect( this->actionIsosurface_mode, SIGNAL( triggered() ), this, SLOT( view_isosurface_mode() ) );
    connect( this->actionSlab_mode_X, SIGNAL( triggered() ), this, SLOT( view_slab_x() ) );
    connect( this->actionSlab_mode_Y, SIGNAL( triggered() ), this, SLOT( view_slab_y() ) );
    connect( this->actionSlab_mode_Z, SIGNAL( triggered() ), this, SLOT( view_slab_z() ) );
    connect( this->actionToggle_visualisation, SIGNAL( triggered() ), this, SLOT( toggleSpinWidget() ) );
    connect( this->actionToggle_large_visualisation, SIGNAL( triggered() ), this, SLOT( view_toggle_spins_only() ) );
    connect( this->actionToggle_fullscreen_window, SIGNAL( triggered() ), this, SLOT( view_toggle_fullscreen() ) );

    // Help Menu
    connect( this->actionKey_Bindings, SIGNAL( triggered() ), this, SLOT( keyBindings() ) );
    connect( this->actionAbout_this_Application, SIGNAL( triggered() ), this, SLOT( about() ) );

    // MenuBar updates
    connect( this->actionShow_Settings, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionShow_Plots, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionShow_Debug, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_Geometry, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_visualisation, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_large_visualisation, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_fullscreen_window, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_infowidget, SIGNAL( triggered() ), this, SLOT( toggleInfoWidget() ) );
    connect( this->actionToggle_Dragging_mode, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_Defect_mode, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->actionToggle_Pinning_mode, SIGNAL( triggered() ), this, SLOT( updateMenuBar() ) );
    connect( this->dockWidget_Settings, SIGNAL( visibilityChanged( bool ) ), this, SLOT( updateMenuBar() ) );
    connect( this->dockWidget_Plots, SIGNAL( visibilityChanged( bool ) ), this, SLOT( updateMenuBar() ) );
    connect( this->dockWidget_Debug, SIGNAL( visibilityChanged( bool ) ), this, SLOT( updateMenuBar() ) );

    // Status Bar
    //      Set a monospace font
    Ui::MainWindow::statusBar->setStyleSheet( "QWidget{font-family: \"Courier\"}" );
    //      Spacer
    this->m_Spacer_5 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Spacer_5 );
    //      Torque
    this->m_Label_Torque = new QLabel( "F_max: -" );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Label_Torque );
    //      Spacer
    this->m_Spacer_4 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Spacer_4 );
    //      Energy
    this->m_Label_E = new QLabel( "E: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Label_E );
    //      M_z
    this->m_Label_Mz = new QLabel( "M_z: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Label_Mz );
    //      Spacer
    this->m_Spacer_3 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Spacer_3 );
    //      FPS
    this->m_Label_FPS = new QLabel( "FPS: -" );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Label_FPS );
    //      Spacer
    this->m_Spacer_2 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Spacer_2 );
    //      N_Cells
    this->m_Label_Dims = new QLabel( "Dims: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_Dims );
    //      Spacer
    this->m_Spacer_1 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( m_Spacer_1 );
    //      NOS
    this->m_Label_NOS = new QLabel( "NOS: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_NOS );
    //      NOI
    this->m_Label_NOI = new QLabel( "NOI: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_NOI );
    //      Initialisations
    this->createStatusBar();
    //      MenuBar checkboxes
    QTimer::singleShot( 1000, this, SLOT( updateMenuBar() ) );

    // Set up Update Timers
    m_timer         = new QTimer( this );
    m_timer_camera  = new QTimer( this );
    m_timer_control = new QTimer( this );
    // m_timer_plots = new QTimer(this);
    // m_timer_spins = new QTimer(this);

    // Connect the Timers
    connect( m_timer, &QTimer::timeout, this, &MainWindow::updateStatusBar );
    connect( m_timer_camera, &QTimer::timeout, this, &MainWindow::move_and_rotate_camera );
    connect( m_timer_control, &QTimer::timeout, this->controlWidget, &ControlWidget::updateData );
    // connect(m_timer_plots, &QTimer::timeout, this->plotsWidget->energyPlot,
    // &PlotWidget::updateData); // this currently resets the user's interaction
    // (movement, zoom) connect(m_timer_spins, &QTimer::timeout, this->spinWidget,
    // &Spin_Widget::updateData);

    // Start Timers
    m_timer->start( 200 );
    m_timer_control->start( 200 );
    // m_timer_camera->start(10);
    // m_timer_plots->start(100);
    // m_timer_spins->start(100);
    // m_timer_debug->start(100);

    this->n_screenshots = 0;
    this->camera_keys_pressed
        = std::map<char, bool>{ { 'w', false }, { 'a', false }, { 's', false }, { 'd', false }, { 'q', false },
                                { 'e', false }, { 't', false }, { 'f', false }, { 'g', false }, { 'h', false } };

    // Status Bar message
    Ui::MainWindow::statusBar->showMessage( tr( "Ready" ), 5000 );
    this->return_focus();
    this->setFocus();
}

void MainWindow::view_toggle_spins_only()
{
    if( this->view_spins_only )
    {
        this->view_spins_only = false;
        Ui::MainWindow::statusBar->showMessage( tr( "Showing UI controls" ), 5000 );

        if( !this->pre_spins_only_settings_hidden )
        {
            dockWidget_Settings->show();
            dockWidget_Settings->topLevelWidget()->resize( pre_spins_only_settings_size );
            dockWidget_Settings->move( pre_spins_only_settings_pos );
        }
        if( !this->pre_spins_only_plots_hidden )
        {
            dockWidget_Plots->show();
            dockWidget_Plots->topLevelWidget()->resize( pre_spins_only_plots_size );
            dockWidget_Plots->move( pre_spins_only_plots_pos );
        }
        if( !this->pre_spins_only_debug_hidden )
        {
            dockWidget_Debug->show();
            dockWidget_Debug->topLevelWidget()->resize( pre_spins_only_debug_size );
            dockWidget_Debug->move( pre_spins_only_debug_pos );
        }
        this->controlWidget->show();
    }
    else
    {
        this->view_spins_only = true;
        Ui::MainWindow::statusBar->showMessage( tr( "Hiding UI controls" ), 5000 );

        this->pre_spins_only_settings_hidden = dockWidget_Settings->isHidden();
        this->pre_spins_only_settings_size   = dockWidget_Settings->topLevelWidget()->size();
        this->pre_spins_only_settings_pos    = dockWidget_Settings->pos();

        this->pre_spins_only_plots_hidden = dockWidget_Plots->isHidden();
        this->pre_spins_only_plots_size   = dockWidget_Plots->topLevelWidget()->size();
        this->pre_spins_only_plots_pos    = dockWidget_Plots->pos();

        this->pre_spins_only_debug_hidden = dockWidget_Debug->isHidden();
        this->pre_spins_only_debug_size   = dockWidget_Debug->topLevelWidget()->size();
        this->pre_spins_only_debug_pos    = dockWidget_Debug->pos();

        this->dockWidget_Settings->hide();
        this->dockWidget_Plots->hide();
        this->dockWidget_Debug->hide();
        this->controlWidget->hide();
    }
}

void MainWindow::view_toggle_fullscreen()
{
    if( this->view_fullscreen )
    {
        this->view_fullscreen = false;
        Ui::MainWindow::statusBar->showMessage( tr( "Switching off fullscreen" ), 5000 );
        this->showMaximized();
    }
    else
    {
        this->view_fullscreen = true;
        Ui::MainWindow::statusBar->showMessage( tr( "Switching to fullscreen" ), 5000 );
        this->windowHandle()->setScreen( qApp->screens().last() );
        this->showFullScreen();
        this->setWindowState( Qt::WindowState::WindowFullScreen );
    }
}

void MainWindow::toggleSpinWidget()
{
    if( this->m_spinWidgetActive )
    {
        this->spinWidget->setSuspended( true );
        this->m_spinWidgetActive = false;
    }
    else
    {
        this->spinWidget->setSuspended( false );
        this->m_spinWidgetActive = true;
    }
}

void MainWindow::toggleInfoWidget()
{
    if( this->m_InfoWidgetActive )
    {
        this->infoWidget->hide();
        Ui::MainWindow::statusBar->show();
        this->m_InfoWidgetActive = false;
    }
    else
    {
        this->infoWidget->show();
        Ui::MainWindow::statusBar->hide();
        this->m_InfoWidgetActive = true;
    }
}

void MainWindow::keyPressEvent( QKeyEvent * k )
{
    // Image index
    auto str_image = []( int idx_img, int noi )
    { return std::string( "Image " + std::to_string( idx_img + 1 ) + "/" + std::to_string( noi ) ); };

    // Key Sequences
    if( k->matches( QKeySequence::Copy ) )
    {
        // Copy the current Spin System
        this->edit_copy();
    }
    else if( k->matches( QKeySequence::Cut ) )
    {
        // Cut the current Spin System from the chain
        this->edit_cut();
    }
    else if( k->matches( QKeySequence::Paste ) )
    {
        // Paste clipboard image to current
        this->edit_paste();
    }

    // Custom Key Sequences (Control)
    else if( k->modifiers() & Qt::ControlModifier )
    {
        switch( k->key() )
        {
            // CTRL+Left - Paste image to left of current image
            case Qt::Key_Left: this->edit_insert_left(); break;

            // CTRL+Right - Paste image to right of current image
            case Qt::Key_Right: this->edit_insert_right(); break;

            // CTRL+F - Fullscreen mode
            case Qt::Key_F:
                if( k->modifiers() & Qt::ShiftModifier )
                {
                    // Actual fullscreen
                    this->view_toggle_fullscreen();
                }
                else
                {
                    // Hide all except SpinWidget, MenuBar and StatusBar
                    this->view_toggle_spins_only();
                }
                break;

            // CTRL+V - Toggle SpinWidget Visibility
            case Qt::Key_V:
                if( k->modifiers() & Qt::ShiftModifier )
                    this->toggleSpinWidget();
                break;

            // CTRL+R - Randomize spins
            case Qt::Key_R: this->control_random(); break;

            // CTRL+N - Add noise
            case Qt::Key_N: this->settingsWidget->configurationAddNoise(); break;

            // CTRL+M - Cycle Method
            case Qt::Key_M:
                this->control_cycle_method();
                break;
                // CTRL+S - Cycle Solver
            case Qt::Key_S: this->control_cycle_solver(); break;
        }
    }

    // Single Keys
    else
    {
        // Movement scaling
        float scale       = 10;
        bool shiftpressed = false;
        if( k->modifiers() & Qt::ShiftModifier )
        {
            scale        = 1;
            shiftpressed = true;
        }

        // Detect visualisation mode cycle
        std::map<SpinWidget::SystemMode, std::string> cycle_name{
            { SpinWidget::SystemMode::CUSTOM, "Custom" }, { SpinWidget::SystemMode::ISOSURFACE, "Isosurface" },
            { SpinWidget::SystemMode::SLAB_X, "X slab" }, { SpinWidget::SystemMode::SLAB_Y, "Y slab" },
            { SpinWidget::SystemMode::SLAB_Z, "Z slab" },
        };

        switch( k->key() )
        {
            // Escape: try to return focus to MainWindow
            case Qt::Key_Escape: this->setFocus(); break;
            // Up: ...
            case Qt::Key_Up:
                Ui::MainWindow::statusBar->showMessage(
                    tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ),
                    5000 );
                break;
            // Left: switch to image left of current image
            case Qt::Key_Left:
                this->controlWidget->prev_image();
                Ui::MainWindow::statusBar->showMessage(
                    tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ),
                    5000 );
                break;
            // Left: switch to image left of current image
            case Qt::Key_Right:
                this->controlWidget->next_image();
                Ui::MainWindow::statusBar->showMessage(
                    tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ),
                    5000 );
                break;
            // Down: ...
            case Qt::Key_Down:
                Ui::MainWindow::statusBar->showMessage(
                    tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ),
                    5000 );
                break;
            // Space: Play and Pause
            case Qt::Key_Space: this->control_playpause(); break;
            // Enter: Insert Configuration
            case Qt::Key_Enter:
            case Qt::Key_Return:
                if( this->hasFocus() || this->spinWidget->hasFocus() )
                    this->control_insertconfiguration();
                break;
            // Display info widget
            case Qt::Key_I: this->toggleInfoWidget(); break;
            // WASDQE
            case Qt::Key_W: this->camera_keys_pressed['w'] = true; break;
            // WASDQE
            case Qt::Key_A: this->camera_keys_pressed['a'] = true; break;
            // WASDQE
            case Qt::Key_S: this->camera_keys_pressed['s'] = true; break;
            // WASDQE
            case Qt::Key_D: this->camera_keys_pressed['d'] = true; break;
            // WASDQE
            case Qt::Key_Q: this->camera_keys_pressed['q'] = true; break;
            // WASDQE
            case Qt::Key_E: this->camera_keys_pressed['e'] = true; break;
            // Movement
            case Qt::Key_T: this->camera_keys_pressed['t'] = true; break;
            // Movement
            case Qt::Key_F: this->camera_keys_pressed['f'] = true; break;
            // Movement
            case Qt::Key_G: this->camera_keys_pressed['g'] = true; break;
            // Movement
            case Qt::Key_H: this->camera_keys_pressed['h'] = true; break;
            // F1: Show key bindings
            case Qt::Key_F1: this->keyBindings(); break;
            // F2: Toggle settings widget
            case Qt::Key_F2: this->view_toggleSettings(); break;
            // F3: Toggle Plots widget
            case Qt::Key_F3: this->view_togglePlots(); break;
            // F4: Toggle debug widget
            case Qt::Key_F4: this->view_toggleDebug(); break;
            // F5: Toggle drag mode
            case Qt::Key_F5:
                this->view_togglePasteMode( SpinWidget::InteractionMode::DRAG );
                this->updateMenuBar();
                break;
            // F6: Toggle defect mode
            case Qt::Key_F6:
                this->view_togglePasteMode( SpinWidget::InteractionMode::DEFECT );
                this->updateMenuBar();
                break;
            // F7: Toggle pinning mode
            case Qt::Key_F7:
                this->view_togglePasteMode( SpinWidget::InteractionMode::PIN );
                this->updateMenuBar();
                break;
            case Qt::Key_Equal:
            case Qt::Key_Plus:
                this->settingsWidget->incrementNCellStep( -1 );
                this->updateStatusBar();
                break;
            case Qt::Key_Minus:
                this->settingsWidget->incrementNCellStep( 1 );
                this->updateStatusBar();
                break;
            // 0: ...
            case Qt::Key_0: break;
            // 1: Custom mode
            case Qt::Key_1:
                this->spinWidget->cycleSystem( SpinWidget::SystemMode::CUSTOM );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage( tr( "Custom mode" ), 5000 );
                break;
            // 2: Isosurface mode
            case Qt::Key_2:
                this->spinWidget->cycleSystem( SpinWidget::SystemMode::ISOSURFACE );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage( tr( "Isosurface mode" ), 5000 );
                break;
            // 3: X slab mode
            case Qt::Key_3:
                this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_X );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage( tr( "X slab mode" ), 5000 );
                break;
            // 4: Y slab mode
            case Qt::Key_4:
                this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_Y );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage( tr( "Y slab mode" ), 5000 );
                break;
            // 5: Z slab mode
            case Qt::Key_5:
                this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_Z );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage( tr( "Z slab mode" ), 5000 );
                break;
            // Delete: Delete current image
            case Qt::Key_Delete: this->edit_delete(); break;
            // Camera
            case Qt::Key_X:
                this->spinWidget->setCameraToX( shiftpressed );
                if( shiftpressed )
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: X view (from back)" ), 5000 );
                else
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: X view (from front)" ), 5000 );
                break;
            case Qt::Key_Y:
                this->spinWidget->setCameraToY( shiftpressed );
                if( shiftpressed )
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: Y view (from back)" ), 5000 );
                else
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: Y view (from front)" ), 5000 );
                break;
            case Qt::Key_Z:
                this->spinWidget->setCameraToZ( shiftpressed );
                if( shiftpressed )
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: Z view (from bottom)" ), 5000 );
                else
                    Ui::MainWindow::statusBar->showMessage( tr( "Camera: Z view (from top)" ), 5000 );
                break;
            case Qt::Key_C: this->view_cycle_camera(); break;
            // Visualisation: cycle and slab
            case Qt::Key_Comma:
            case Qt::Key_Less:
            case Qt::Key_Semicolon:
                this->spinWidget->moveSlab( -10.0 / scale );
                this->settingsWidget->updateData();
                break;
            case Qt::Key_Period:
            case Qt::Key_Greater:
            case Qt::Key_Colon:
                this->spinWidget->moveSlab( 10.0 / scale );
                this->settingsWidget->updateData();
                break;
            case Qt::Key_Slash:
            case Qt::Key_Question:
                this->spinWidget->cycleSystem( !shiftpressed );
                this->settingsWidget->updateData();
                Ui::MainWindow::statusBar->showMessage(
                    tr( ( "Cycled mode to " + cycle_name[spinWidget->systemCycle()] ).c_str() ), 5000 );
                break;
            case Qt::Key_F10: this->view_toggle_spins_only(); break;
            case Qt::Key_F11: this->view_toggle_fullscreen(); break;
            case Qt::Key_Home:
            case Qt::Key_F12: this->takeScreenshot(); break;
        }
    }
    this->return_focus();

    if( std::any_of(
            this->camera_keys_pressed.begin(), this->camera_keys_pressed.end(),
            []( const std::pair<char, bool> & key ) -> bool { return key.second; } ) )
        m_timer_camera->start( 10 );
}

void MainWindow::keyReleaseEvent( QKeyEvent * k )
{
    switch( k->key() )
    {
        // WASDQE
        case Qt::Key_W: this->camera_keys_pressed['w'] = false; break;
        // WASDQE
        case Qt::Key_A: this->camera_keys_pressed['a'] = false; break;
        // WASDQE
        case Qt::Key_S: this->camera_keys_pressed['s'] = false; break;
        // WASDQE
        case Qt::Key_D: this->camera_keys_pressed['d'] = false; break;
        // WASDQE
        case Qt::Key_Q: this->camera_keys_pressed['q'] = false; break;
        // WASDQE
        case Qt::Key_E: this->camera_keys_pressed['e'] = false; break;
        // Movement
        case Qt::Key_T: this->camera_keys_pressed['t'] = false; break;
        // Movement
        case Qt::Key_F: this->camera_keys_pressed['f'] = false; break;
        // Movement
        case Qt::Key_G: this->camera_keys_pressed['g'] = false; break;
        // Movement
        case Qt::Key_H: this->camera_keys_pressed['h'] = false; break;
    }

    // Stop the camera update timer if all camera keys were released
    if( !std::any_of(
            this->camera_keys_pressed.begin(), this->camera_keys_pressed.end(),
            []( const std::pair<char, bool> & key ) -> bool { return key.second; } ) )
        m_timer_camera->stop();
}

void MainWindow::move_and_rotate_camera()
{
    int w = 0, a = 0, s = 0, d = 0;
    int q = 0, e = 0;
    int t = 0, f = 0, g = 0, h = 0;

    float scale       = 5;
    bool shiftpressed = false;

    if( QApplication::keyboardModifiers() & Qt::ShiftModifier )
    {
        scale        = 1;
        shiftpressed = true;
    }

    w = (int)this->camera_keys_pressed['w'];
    a = (int)this->camera_keys_pressed['a'];
    s = (int)this->camera_keys_pressed['s'];
    d = (int)this->camera_keys_pressed['d'];
    q = (int)this->camera_keys_pressed['q'];
    e = (int)this->camera_keys_pressed['e'];
    t = (int)this->camera_keys_pressed['t'];
    f = (int)this->camera_keys_pressed['f'];
    g = (int)this->camera_keys_pressed['g'];
    h = (int)this->camera_keys_pressed['h'];

    this->spinWidget->moveCamera(
        2 * scale * s - 2 * scale * w, 2 * scale * f - 2 * scale * h, 2 * scale * t - 2 * scale * g );
    this->spinWidget->rotateCamera( 2 * scale * q - 2 * scale * e, 2 * scale * a - 2 * scale * d );
}

void MainWindow::view_toggleDebug()
{
    if( this->dockWidget_Debug->isVisible() )
        this->dockWidget_Debug->hide();
    else
        this->dockWidget_Debug->show();
}

void MainWindow::view_toggleGeometry()
{
    this->settingsWidget->toggleGeometry();
}

void MainWindow::view_togglePlots()
{
    if( this->dockWidget_Plots->isVisible() )
        this->dockWidget_Plots->hide();
    else
        this->dockWidget_Plots->show();
}

void MainWindow::view_toggleSettings()
{
    if( this->dockWidget_Settings->isVisible() )
        this->dockWidget_Settings->hide();
    else
        this->dockWidget_Settings->show();
}

void MainWindow::view_togglePasteMode( SpinWidget::InteractionMode mode )
{
    if( this->spinWidget->interactionMode() == SpinWidget::InteractionMode::DRAG
        && mode == SpinWidget::InteractionMode::DRAG )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::REGULAR );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Regular" ), 5000 );
    }
    else if(
        this->spinWidget->interactionMode() == SpinWidget::InteractionMode::DEFECT
        && mode == SpinWidget::InteractionMode::DEFECT )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::REGULAR );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Regular" ), 5000 );
    }
    else if(
        this->spinWidget->interactionMode() == SpinWidget::InteractionMode::PIN
        && mode == SpinWidget::InteractionMode::PIN )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::REGULAR );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Regular" ), 5000 );
    }
    else if( mode == SpinWidget::InteractionMode::DRAG )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::DRAG );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Drag" ), 5000 );
    }
    else if( mode == SpinWidget::InteractionMode::DEFECT )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::DEFECT );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Defect" ), 5000 );
    }
    else if( mode == SpinWidget::InteractionMode::PIN )
    {
        this->spinWidget->setInteractionMode( SpinWidget::InteractionMode::PIN );
        Ui::MainWindow::statusBar->showMessage( tr( "Interaction Mode: Pinning" ), 5000 );
    }
    this->settingsWidget->updateData();
}

void MainWindow::createStatusBar()
{
    // Remove previous IPS labels
    for( unsigned int i = 0; i < this->m_Labels_IPS.size(); ++i )
    {
        Ui::MainWindow::statusBar->removeWidget( this->m_Labels_IPS[i] );
    }
    // Remove Spacers and Torque
    Ui::MainWindow::statusBar->removeWidget( this->m_Spacer_5 );
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_Torque );
    Ui::MainWindow::statusBar->removeWidget( this->m_Spacer_4 );

    // Create IPS Labels and add them to the statusBar
    this->m_Labels_IPS = std::vector<QLabel *>( 0 );
    if( Simulation_Running_Anywhere_On_Chain( state.get() ) )
    {
        if( Simulation_Running_On_Chain( state.get() ) )
        {
            this->m_Labels_IPS.push_back( new QLabel );
            this->m_Labels_IPS.back()->setText( "IPS[-]: -  " );
            Ui::MainWindow::statusBar->addPermanentWidget( m_Labels_IPS.back() );
        }
        else
        {
            for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
            {
                if( Simulation_Running_On_Image( state.get(), img ) )
                {
                    this->m_Labels_IPS.push_back( new QLabel );
                    this->m_Labels_IPS.back()->setText( "IPS[-]: -  " );
                    Ui::MainWindow::statusBar->addPermanentWidget( m_Labels_IPS.back() );
                }
            }
        }
        //      Spacer
        this->m_Spacer_5 = new QLabel( "|  " );
        Ui::MainWindow::statusBar->addPermanentWidget( this->m_Spacer_5 );
        //      Torque
        this->m_Label_Torque = new QLabel( "F_max: -" );
        Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_Torque );
        //      Spacer
        this->m_Spacer_4 = new QLabel( "  |  " );
        Ui::MainWindow::statusBar->addPermanentWidget( this->m_Spacer_4 );
    }

    //      Energy
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_E );
    this->m_Label_E = new QLabel( "E: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_E );

    //      M_z
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_Mz );
    this->m_Label_Mz = new QLabel;
    this->m_Label_Mz->setText( "M_z: -  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_Mz );

    //      Spacer
    Ui::MainWindow::statusBar->removeWidget( this->m_Spacer_3 );
    this->m_Spacer_3 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Spacer_3 );

    //      FPS
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_FPS );
    this->m_Label_FPS = new QLabel( "FPS: -" );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_FPS );

    //      Spacer
    Ui::MainWindow::statusBar->removeWidget( this->m_Spacer_2 );
    this->m_Spacer_2 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Spacer_2 );

    //      Dims
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_Dims );
    this->m_Label_Dims = new QLabel;
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int nth      = this->spinWidget->visualisationNCellSteps();
    QString text = QString::fromLatin1( "Dims: " ) + QString::number( n_cells[0] ) + QString::fromLatin1( "x" )
                   + QString::number( n_cells[1] ) + QString::fromLatin1( "x" ) + QString::number( n_cells[2] );
    if( nth == 2 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "nd)" );
    }
    else if( nth == 3 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "rd)" );
    }
    else if( nth > 3 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "th)" );
    }
    this->m_Label_Dims->setText( text );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_Dims );

    //      Spacer
    Ui::MainWindow::statusBar->removeWidget( this->m_Spacer_1 );
    this->m_Spacer_1 = new QLabel( "  |  " );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Spacer_1 );

    //      NOS
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_NOS );
    this->m_Label_NOS = new QLabel;
    int nos           = System_Get_NOS( this->state.get() );
    QString nosqstring;
    if( nos < 1e5 )
        nosqstring = QString::number( nos );
    else
        nosqstring = QString::number( (float)nos, 'E', 2 );
    this->m_Label_NOS->setText( QString::fromLatin1( "NOS: " ) + nosqstring + QString::fromLatin1( " " ) );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_NOS );

    //      NOI
    Ui::MainWindow::statusBar->removeWidget( this->m_Label_NOI );
    this->m_Label_NOI = new QLabel;
    this->m_Label_NOI->setText(
        QString::fromLatin1( "NOI: " ) + QString::number( Chain_Get_NOI( this->state.get() ) )
        + QString::fromLatin1( " " ) );
    Ui::MainWindow::statusBar->addPermanentWidget( this->m_Label_NOI );

    // Update contents
    this->updateStatusBar();
}

void MainWindow::updateStatusBar()
{
    this->m_Label_FPS->setText(
        QString::fromLatin1( "FPS: " ) + QString::number( (int)this->spinWidget->getFramesPerSecond() ) );

    float F = Simulation_Get_MaxTorqueComponent( state.get() );
    if( !Simulation_Running_On_Chain( state.get() ) )
        this->m_Label_Torque->setText( QString::fromLatin1( "F_max: " ) + QString::number( F, 'E', 2 ) );

    double E = System_Get_Energy( state.get() ) / System_Get_NOS( state.get() );
    this->m_Label_E->setText(
        QString::fromLatin1( "E: " ) + QString::number( E, 'f', 6 ) + QString::fromLatin1( "  " ) );

    float M[3];
    Quantity_Get_Magnetization( state.get(), M );
    this->m_Label_Mz->setText( QString::fromLatin1( "M_z: " ) + QString::number( M[2], 'f', 6 ) );

    float ips;
    int precision;
    QString qstr_ips;
    std::vector<QString> v_str( 0 );

    if( Simulation_Running_On_Chain( state.get() ) )
    {
        float * f = new float[Chain_Get_NOI( state.get() )];
        Simulation_Get_Chain_MaxTorqueComponents( state.get(), f );
        float f_current = f[System_Get_Index( state.get() )];
        this->m_Label_Torque->setText(
            QString::fromLatin1( "F_current: " ) + QString::number( f_current, 'E', 2 )
            + QString::fromLatin1( "  F_max: " ) + QString::number( F, 'E', 2 ) );
        ips = Simulation_Get_IterationsPerSecond( state.get(), -1 );
        if( ips < 1 )
            precision = 4;
        else if( ips > 99 )
            precision = 0;
        else
            precision = 2;
        if( ips < 1e5 )
            qstr_ips = QString::number( ips, 'f', precision );
        else
            qstr_ips = QString::fromLatin1( "> 100k" );
        v_str.push_back( QString::fromLatin1( "IPS: " ) + qstr_ips + QString::fromLatin1( "  " ) );
    }
    else
    {
        for( int img = 0; img < Chain_Get_NOI( state.get() ); ++img )
        {
            if( Simulation_Running_On_Image( state.get(), img ) )
            {
                ips = Simulation_Get_IterationsPerSecond( state.get(), img );
                if( ips < 1 )
                    precision = 4;
                else if( ips > 99 )
                    precision = 0;
                else
                    precision = 2;
                if( ips < 1e5 )
                    qstr_ips = QString::number( ips, 'f', precision );
                else
                    qstr_ips = QString::fromLatin1( "> 100k" );
                v_str.push_back(
                    QString::fromLatin1( "IPS[" ) + QString::number( img + 1 ) + QString::fromLatin1( "]: " ) + qstr_ips
                    + QString::fromLatin1( "  " ) );
            }
        }
    }

    if( v_str.size() != m_Labels_IPS.size() )
        createStatusBar();
    else
    {
        for( unsigned int i = 0; i < m_Labels_IPS.size() && i < v_str.size(); ++i )
        {
            this->m_Labels_IPS[i]->setText( v_str[i] );
        }
    }

    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int nth      = this->spinWidget->visualisationNCellSteps();
    QString text = QString::fromLatin1( "Dims: " ) + QString::number( n_cells[0] ) + QString::fromLatin1( "x" )
                   + QString::number( n_cells[1] ) + QString::fromLatin1( "x" ) + QString::number( n_cells[2] );
    if( nth == 2 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "nd)" );
    }
    else if( nth == 3 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "rd)" );
    }
    else if( nth > 3 )
    {
        text += QString::fromLatin1( "    (using every " ) + QString::number( nth ) + QString::fromLatin1( "th)" );
    }
    this->m_Label_Dims->setText( text );
}

void MainWindow::updateMenuBar()
{
    // Settings
    if( this->dockWidget_Settings->isVisible() )
        this->actionShow_Settings->setChecked( true );
    else
        this->actionShow_Settings->setChecked( false );
    // Plots
    if( this->dockWidget_Plots->isVisible() )
        this->actionShow_Plots->setChecked( true );
    else
        this->actionShow_Plots->setChecked( false );
    // Debug
    if( this->dockWidget_Debug->isVisible() )
        this->actionShow_Debug->setChecked( true );
    else
        this->actionShow_Debug->setChecked( false );

    // Controls
    this->actionToggle_Dragging_mode->setChecked(
        this->spinWidget->interactionMode() == SpinWidget::InteractionMode::DRAG );
    this->actionToggle_Defect_mode->setChecked(
        this->spinWidget->interactionMode() == SpinWidget::InteractionMode::DEFECT );
    this->actionToggle_Pinning_mode->setChecked(
        this->spinWidget->interactionMode() == SpinWidget::InteractionMode::PIN );

    // Info Widget
    this->actionToggle_infowidget->setChecked( this->m_InfoWidgetActive );

    // Visualisation
    this->actionToggle_visualisation->setChecked( this->m_spinWidgetActive );
    this->actionToggle_large_visualisation->setChecked( this->view_spins_only );
    this->actionToggle_fullscreen_window->setChecked( this->view_fullscreen );
}

void MainWindow::takeScreenshot()
{
    std::string tag = State_DateTime( state.get() );
    ++n_screenshots;
    std::string name = tag + "_Screenshot_" + std::to_string( n_screenshots );
    this->spinWidget->screenShot( name );
    Ui::MainWindow::statusBar->showMessage( tr( ( "Made Screenshot " + name ).c_str() ), 5000 );
}

void MainWindow::take_Screenshot_Chain()
{
    int active_image = System_Get_Index( this->state.get() );
    State_DateTime( state.get() );
    Chain_Jump_To_Image( this->state.get(), 0 );
    std::string tag = State_DateTime( state.get() );
    ++n_screenshots_chain;
    for( int i = 0; i < Chain_Get_NOI( this->state.get() ); ++i )
    {
        std::string name
            = tag + "_Screenshot_Chain_" + std::to_string( n_screenshots_chain ) + "_Image_" + std::to_string( i );
        this->spinWidget->screenShot( name );
        Ui::MainWindow::statusBar->showMessage( tr( ( "Made Screenshot " + name ).c_str() ), 5000 );
        this->controlWidget->next_image();
    }
    Chain_Jump_To_Image( this->state.get(), active_image );
}

void MainWindow::edit_cut()
{
    auto str_image = []( int idx_img, int noi )
    { return std::string( "Image " + std::to_string( idx_img + 1 ) + "/" + std::to_string( noi ) ); };

    this->controlWidget->cut_image();
    Ui::MainWindow::statusBar->showMessage(
        tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ), 5000 );
    this->createStatusBar();
}

void MainWindow::edit_copy()
{
    Chain_Image_to_Clipboard( state.get() );
}

void MainWindow::edit_paste()
{
    this->controlWidget->paste_image();
    this->createStatusBar();
}

void MainWindow::edit_insert_right()
{
    auto str_image = []( int idx_img, int noi )
    { return std::string( "Image " + std::to_string( idx_img + 1 ) + "/" + std::to_string( noi ) ); };

    this->controlWidget->paste_image( "right" );
    Ui::MainWindow::statusBar->showMessage(
        tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ), 5000 );
    this->createStatusBar();
}

void MainWindow::edit_insert_left()
{
    auto str_image = []( int idx_img, int noi )
    { return std::string( "Image " + std::to_string( idx_img + 1 ) + "/" + std::to_string( noi ) ); };

    this->controlWidget->paste_image( "left" );
    Ui::MainWindow::statusBar->showMessage(
        tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ), 5000 );
    this->createStatusBar();
}

void MainWindow::edit_delete()
{
    auto str_image = []( int idx_img, int noi )
    { return std::string( "Image " + std::to_string( idx_img + 1 ) + "/" + std::to_string( noi ) ); };

    this->controlWidget->delete_image();
    Ui::MainWindow::statusBar->showMessage(
        tr( str_image( System_Get_Index( state.get() ), Chain_Get_NOI( this->state.get() ) ).c_str() ), 5000 );
    this->createStatusBar();
}

void MainWindow::control_random()
{
    this->settingsWidget->randomPressed();
}

void MainWindow::control_insertconfiguration()
{
    this->settingsWidget->lastConfiguration();
}

void MainWindow::control_playpause()
{
    this->controlWidget->play_pause();
    Ui::MainWindow::statusBar->showMessage(
        tr( std::string( "Play/Pause: " + this->controlWidget->methodName() + " simulation" ).c_str() ), 5000 );
}

void MainWindow::control_cycle_method()
{
    this->controlWidget->cycleMethod();
    Ui::MainWindow::statusBar->showMessage( tr( this->controlWidget->methodName().c_str() ), 5000 );
}

void MainWindow::control_cycle_solver()
{
    this->controlWidget->cycleSolver();
    Ui::MainWindow::statusBar->showMessage( tr( this->controlWidget->solverName().c_str() ), 5000 );
}

void MainWindow::view_regular_mode()
{
    this->spinWidget->cycleSystem( SpinWidget::SystemMode::CUSTOM );
    this->settingsWidget->updateData();
    Ui::MainWindow::statusBar->showMessage( tr( "Set mode to Regular" ), 5000 );
}

void MainWindow::view_isosurface_mode()
{
    this->spinWidget->cycleSystem( SpinWidget::SystemMode::ISOSURFACE );
    this->settingsWidget->updateData();
    Ui::MainWindow::statusBar->showMessage( tr( "Set mode to Isosurface" ), 5000 );
}

void MainWindow::view_slab_x()
{
    this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_X );
    this->settingsWidget->updateData();
    Ui::MainWindow::statusBar->showMessage( tr( "Set mode to Slab X" ), 5000 );
}

void MainWindow::view_slab_y()
{
    this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_Y );
    this->settingsWidget->updateData();
    Ui::MainWindow::statusBar->showMessage( tr( "Set mode to Slab Y" ), 5000 );
}

void MainWindow::view_slab_z()
{
    this->spinWidget->cycleSystem( SpinWidget::SystemMode::SLAB_Z );
    this->settingsWidget->updateData();
    Ui::MainWindow::statusBar->showMessage( tr( "Set mode to Slab Z" ), 5000 );
}

void MainWindow::view_cycle_camera()
{
    this->spinWidget->cycleCamera();
    if( this->spinWidget->cameraProjection() )
        Ui::MainWindow::statusBar->showMessage( tr( "Camera: perspective projection" ), 5000 );
    else
        Ui::MainWindow::statusBar->showMessage( tr( "Camera: orthogonal projection" ), 5000 );
    this->settingsWidget->updateData();
}

void MainWindow::about()
{
    QMessageBox about;
    about.setInformativeText(
        QString::fromLatin1( "Library version " ) + Spirit_Version_Full()
        + QString::fromLatin1( "<br><br>"
                               "The <b>Spirit</b> GUI application incorporates intuitive "
                               "visualisation,<br>"
                               "powerful <b>Spin Dynamics</b> and <b>Nudged Elastic Band</b> tools<br>"
                               "into a cross-platform user interface.<br>"
                               "<br>"
                               "Main developers:<br>"
                               "  - Gideon Mueller (<a "
                               "href=\"mailto:g.mueller@fz-juelich.de\">g.mueller@fz-juelich.de</a>)<br>"
                               "  - Moritz Sallermann (<a "
                               "href=\"mailto:m.sallermann@fz-juelich.de\">m.sallermann@fz-juelich.de</"
                               "a>)<br>"
                               "at the Institute for Advanced Simulation 1 of the Forschungszentrum "
                               "Juelich.<br>"
                               "For more information about us, visit <a "
                               "href=\"http://juspin.de\">juSpin.de</a><br>"
                               "or see the <a "
                               "href=\"http://www.fz-juelich.de/pgi/pgi-1/DE/Home/home_node.html\">IAS-1 "
                               "Website</a>.<br>"
                               "<br>"
                               "The sources are hosted at <a "
                               "href=\"https://spirit-code.github.io\">spirit-code.github.io</a>"
                               " and the documentation can be found at <a "
                               "href=\"https://spirit-docs.readthedocs.io\">spirit-docs.readthedocs.io</"
                               "a>.<br>" ) );
    about.setStandardButtons( QMessageBox::Close );
    about.setDefaultButton( QMessageBox::Close );
    about.setIconPixmap(
        QPixmap( ":/Icons/Logo_Ghost" ).scaled( QSize( 256, 256 ), Qt::KeepAspectRatio, Qt::SmoothTransformation ) );
    about.exec();
}

void MainWindow::keyBindings()
{
    QMessageBox::about(
        this, tr( "Spirit UI Key Bindings" ),
        QString::fromLatin1( "The <b>Key Bindings</b> are as follows:<br>"
                             "<br>"
                             "<i>UI controls</i><br>"
                             " - <b>F1</b>:      Show this<br>"
                             " - <b>F2</b>:      Toggle Settings<br>"
                             " - <b>F3</b>:      Toggle Plots<br>"
                             " - <b>F4</b>:      Toggle Debug<br>"
                             " - <b>F5</b>:      Toggle \"Dragging\" mode<br>"
                             " - <b>F6</b>:      Toggle \"Defects\" mode<br>"
                             " - <b>F7</b>:      Toggle \"Pinning\" mode<br>"
                             " - <b>F10 and Ctrl+F</b>:        Toggle large visualisation<br>"
                             " - <b>F11 and Ctrl+Shift+F</b>:  Toggle fullscreen window<br>"
                             " - <b>F12 and Home</b>:          Screenshot of Visualization region<br>"
                             " - <b>Ctrl+Shift+V</b>:          Toggle OpenGL Visualisation<br>"
                             " - <b>i</b>:       Toggle large visualisation<br>"
                             " - <b>Escape</b>:  Try to return focus to main UI (does not always work)<br>"
                             "<br>"
                             "<i>Camera controls</i><br>"
                             " - <b>Left mouse</b>:    Rotate the camera around (<b>shift</b> to go "
                             "slow)<br>"
                             " - <b>Right mouse</b>:   Move the camera around (<b>shift</b> to go "
                             "slow)<br>"
                             " - <b>Scroll mouse</b>:  Zoom in on focus point (<b>shift</b> to go "
                             "slow)<br>"
                             " - <b>WASD</b>:    Rotate the camera around (<b>shift</b> to go slow)<br>"
                             " - <b>TFGH</b>:    Move the camera around (<b>shift</b> to go slow)<br>"
                             " - <b>X,Y,Z</b>:   Set the camera in X, Y or Z direction (<b>shift</b> to "
                             "invert)<br>"
                             "<br>"
                             "<i>Control Simulations</i><br>"
                             " - <b>Space</b>:   Play/Pause<br>"
                             " - <b>Ctrl+M</b>:  Cycle Method<br>"
                             " - <b>Ctrl+S</b>:  Cycle Solver<br>"
                             "<br>"
                             "<i>Manipulate the current images</i><br>"
                             " - <b>Ctrl+R</b>:  Random configuration<br>"
                             " - <b>Ctrl+N</b>:  Add tempered noise<br>"
                             " - <b>Enter</b>:   Insert last used configuration<br>"
                             "<br>"
                             "<i>Visualisation</i><br>"
                             " - <b>+/-</b>:     Use more/fewer data points of the vector field<br>"
                             " - <b>1</b>:       Regular Visualisation Mode<br>"
                             " - <b>2</b>:       Isosurface Visualisation Mode<br>"
                             " - <b>3-5</b>:     Slab (X,Y,Z) Visualisation Mode<br>"
                             " - <b>/</b>:       Cycle Visualisation Mode<br>"
                             " - <b>, and .</b>: Move Slab (<b>shift</b> to go faster)<br>"
                             "<br>"
                             "<i>Manipulate the chain of images</i><br>"
                             " - <b>Arrows</b>:           Switch between images and chains<br>"
                             " - <b>Ctrl+X</b>:           Cut   image<br>"
                             " - <b>Ctrl+C</b>:           Copy  image<br>"
                             " - <b>Ctrl+V</b>:           Paste image at current index<br>"
                             " - <b>Ctrl+Left/Right</b>:  Insert left/right of current index<br>"
                             " - <b>Del</b>:              Delete image<br>"
                             "<br>"
                             "<i>Note that some of the keybindings may only work correctly on US keyboard "
                             "layout.</i><br>"
                             "<br>"
                             "For more information see the documentation at <a "
                             "href=\"https://spirit-docs.readthedocs.io\">spirit-docs.readthedocs.io</"
                             "a>" ) );
}

void MainWindow::return_focus()
{
    auto childWidgets = this->findChildren<QWidget *>();
    for( int i = 0; i < childWidgets.count(); ++i )
    {
        childWidgets.at( i )->clearFocus();
    }
}

void MainWindow::save_Spin_Configuration()
{
    QString selectedFilter;
    auto fileName = QFileDialog::getSaveFileName(
        this, tr( "Save Spin Configuration" ), "./output",
        tr( "Any (*);;OOMF Vector Field text (*.ovf);;"
            "OOMF Vector Field binary (*.ovf);;OOMF Vector Field binary 8 (*.ovf);;"
            "OOMF Vector Field binary 4 (*.ovf);;OOMF Vector Field csv (*.ovf)" ),
        &selectedFilter );

    if( !fileName.isEmpty() )
    {
        int fileFormat = IO_Fileformat_OVF_text;
        if( selectedFilter == "OOMF Vector Field binary (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin;
        if( selectedFilter == "OOMF Vector Field binary 4 (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin4;
        if( selectedFilter == "OOMF Vector Field binary 8 (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin8;
        if( selectedFilter == "OOMF Vector Field csv (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_csv;
        auto file = string_q2std( fileName );
        IO_Image_Write( this->state.get(), file.c_str(), fileFormat );
    }
}

void MainWindow::load_Spin_Configuration()
{
    auto fileNames = QFileDialog::getOpenFileNames(
        this, tr( "Load Spin Configuration" ), "./input",
        tr( "Any (*);;OOMF Vector Field (*.ovf);;"
            "Plaintext (*.txt);;Comma-separated (*.csv)" ) );

    int n_files       = fileNames.size();
    int n_images_read = 0;
    int i_start       = System_Get_Index( this->state.get() );

    for( auto & fileName : fileNames )
    {
        int noi         = Chain_Get_NOI( this->state.get() );
        bool read_image = true;

        if( fileName.isEmpty() )
            read_image = false;

        if( read_image )
        {
            // Append image to chain if necessary
            if( i_start + n_images_read == noi )
            {
                Chain_Image_to_Clipboard( this->state.get() );
                Chain_Insert_Image_After( this->state.get() );
                Chain_next_Image( this->state.get() );
            }
            else if( n_images_read > 0 )
            {
                Chain_next_Image( this->state.get() );
            }

            // Read the file
            auto file = string_q2std( fileName );
            IO_Image_Read( this->state.get(), file.c_str() );

            ++n_images_read;
        }
    }

    Chain_Jump_To_Image( this->state.get(), i_start );
    this->spinWidget->updateData();
}

void MainWindow::save_Spin_Configuration_Eigenmodes()
{
    QString selectedFilter;
    auto fileName = QFileDialog::getSaveFileName(
        this, tr( "Save Spin Configuration Eigenmodes" ), "./output",
        tr( "Any (*);;OOMF Vector Field text (*.ovf);;"
            "OOMF Vector Field binary (*.ovf);;OOMF Vector Field binary 8 (*.ovf);;"
            "OOMF Vector Field binary 4 (*.ovf);;OOMF Vector Field csv (*.ovf)" ),
        &selectedFilter );

    int fileFormat = IO_Fileformat_OVF_text;
    if( selectedFilter == "OOMF Vector Field binary (*.ovf)" )
        fileFormat = IO_Fileformat_OVF_bin8;
    if( selectedFilter == "OOMF Vector Field binary 8 (*.ovf)" )
        fileFormat = IO_Fileformat_OVF_bin8;
    if( selectedFilter == "OOMF Vector Field binary 4 (*.ovf)" )
        fileFormat = IO_Fileformat_OVF_bin4;
    if( selectedFilter == "OOMF Vector Field csv (*.ovf)" )
        fileFormat = IO_Fileformat_OVF_csv;

    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Eigenmodes_Write( this->state.get(), file.c_str(), fileFormat );
    }
}

void MainWindow::load_Spin_Configuration_Eigenmodes()
{
    auto fileName = QFileDialog::getOpenFileName(
        this, tr( "Load Spin Configuration" ), "./input", tr( "OOMMF Vector Field (*.ovf)" ) );

    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Eigenmodes_Read( this->state.get(), file.c_str() );
    }
}

void MainWindow::save_Spin_Configuration_Chain()
{
    QString selectedFilter;
    auto fileName = QFileDialog::getSaveFileName(
        this, tr( "Save SpinChain Configuration" ), "./output",
        tr( "Any (*);;OOMF Vector Field text (*.ovf);;"
            "OOMF Vector Field binary (*.ovf);;OOMF Vector Field binary 8 (*.ovf);;"
            "OOMF Vector Field binary 4 (*.ovf);;OOMF Vector Field csv (*.ovf)" ),
        &selectedFilter );

    if( !fileName.isEmpty() )
    {
        int fileFormat = IO_Fileformat_OVF_text;
        if( selectedFilter == "OOMF Vector Field binary (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin8;
        if( selectedFilter == "OOMF Vector Field binary 8 (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin8;
        if( selectedFilter == "OOMF Vector Field binary 4 (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_bin4;
        if( selectedFilter == "OOMF Vector Field csv (*.ovf)" )
            fileFormat = IO_Fileformat_OVF_csv;
        auto file = string_q2std( fileName );
        IO_Chain_Write( this->state.get(), file.c_str(), fileFormat );
    }
}

void MainWindow::load_Spin_Configuration_Chain()
{
    auto fileName = QFileDialog::getOpenFileName(
        this, tr( "Load Spin Configuration" ), "./input",
        tr( "Any (*);;OOMF Vector Field (*.ovf);;"
            "Plaintext (*.txt);;Comma-separated (*.csv)" ) );

    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Chain_Read( this->state.get(), file.c_str() );
    }
    this->createStatusBar();
    this->controlWidget->updateData();
    this->spinWidget->updateData();
}

void MainWindow::load_Configuration()
{
    int idx_img = System_Get_Index( state.get() );
    // Read Spin System from cfg
    auto fileName = QFileDialog::getOpenFileName( this, tr( "Open Config" ), "./input", tr( "Config (*.cfg)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );

        // Set current image
        if( !IO_System_From_Config( this->state.get(), file.c_str() ) )
        {
            QMessageBox::about(
                this, tr( "Error" ),
                tr( "The given config file could not be used!\n"
                    "Please check the log for any details.\n"
                    "\n"
                    "The system has not been reset!" ) );
        }
        else
        {
            this->updateStatusBar();
            this->spinWidget->updateData();
            this->settingsWidget->updateData();
        }
    }
}

void MainWindow::save_Configuration()
{
    int idx_img = System_Get_Index( state.get() );
    // Read Spin System from cfg
    auto fileName = QFileDialog::getSaveFileName( this, tr( "Save Config" ), "./input", tr( "Config (*.cfg)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        State_To_Config( this->state.get(), file.c_str(), "" );
    }
}

void MainWindow::save_System_Energy_Spins()
{
    this->return_focus();
    auto fileName
        = QFileDialog::getSaveFileName( this, tr( "Save Energies per Spin" ), "./output", tr( "Text (*.txt)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Image_Write_Energy_per_Spin( this->state.get(), file.c_str(), IO_Fileformat_OVF_text );
    }
}

void MainWindow::save_Chain_Energies()
{
    this->return_focus();
    auto fileName = QFileDialog::getSaveFileName( this, tr( "Save Energies" ), "./output", tr( "Text (*.txt)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Chain_Write_Energies( this->state.get(), file.c_str() );
    }
}

void MainWindow::save_Chain_Energies_Interpolated()
{
    this->return_focus();
    auto fileName = QFileDialog::getSaveFileName( this, tr( "Save Energies" ), "./output", tr( "Text (*.txt)" ) );
    if( !fileName.isEmpty() )
    {
        auto file = string_q2std( fileName );
        IO_Chain_Write_Energies_Interpolated( this->state.get(), file.c_str() );
    }
}

void MainWindow::readSettings()
{
    QSettings settings( "Spirit Code", "Spirit" );
    restoreGeometry( settings.value( "geometry" ).toByteArray() );
    restoreState( settings.value( "windowState", QByteArray() ).toByteArray() );
    bool spins_only = settings.value( "fullscreenSpins" ).toBool();
    if( spins_only )
        this->view_toggle_spins_only();
    bool fullscreen = settings.value( "fullscreen" ).toBool();
    if( fullscreen )
        this->view_toggle_fullscreen();
    bool show_infoWidget = settings.value( "show infoWidget", true ).toBool();
    if( !show_infoWidget )
        this->toggleInfoWidget();

    // Settings Dock
    settings.beginGroup( "SettingsDock" );
    // dockWidget_Settings->setFloating(settings.value("docked").toBool());
    // addDockWidget((Qt::DockWidgetArea)settings.value("dockarea",
    // Qt::RightDockWidgetArea).toInt(), dockWidget_Settings);
    // dockWidget_Settings->setHidden(settings.value("hidden").toBool());
    dockWidget_Settings->topLevelWidget()->resize( settings.value( "size", QSize( 1, 1 ) ).toSize() );
    dockWidget_Settings->move( settings.value( "pos", QPoint( 200, 200 ) ).toPoint() );
    settings.endGroup();

    // Plots Dock
    settings.beginGroup( "PlotsDock" );
    // dockWidget_Plots->setFloating(settings.value("docked").toBool());
    // addDockWidget((Qt::DockWidgetArea)settings.value("dockarea",
    // Qt::RightDockWidgetArea).toInt(), dockWidget_Plots);
    // dockWidget_Plots->setHidden(settings.value("hidden").toBool());
    dockWidget_Plots->topLevelWidget()->resize( settings.value( "size", QSize( 1, 1 ) ).toSize() );
    dockWidget_Plots->move( settings.value( "pos", QPoint( 200, 200 ) ).toPoint() );
    settings.endGroup();

    // Debug Dock
    settings.beginGroup( "DebugDock" );
    // dockWidget_Debug->setFloating(settings.value("docked").toBool());
    // addDockWidget((Qt::DockWidgetArea)settings.value("dockarea",
    // Qt::RightDockWidgetArea).toInt(), dockWidget_Debug);
    // dockWidget_Debug->setHidden(settings.value("hidden").toBool());
    dockWidget_Debug->topLevelWidget()->resize( settings.value( "size", QSize( 1, 1 ) ).toSize() );
    dockWidget_Debug->move( settings.value( "pos", QPoint( 200, 200 ) ).toPoint() );
    settings.endGroup();
}

void MainWindow::writeSettings()
{
    QSettings settings( "Spirit Code", "Spirit" );
    settings.setValue( "geometry", saveGeometry() );
    settings.setValue( "windowState", saveState() );
    settings.setValue( "fullscreenSpins", this->view_spins_only );
    settings.setValue( "fullscreen", this->view_fullscreen );
    settings.setValue( "show infoWidget", this->m_InfoWidgetActive );

    // Settings Dock
    settings.beginGroup( "SettingsDock" );
    // settings.setValue("dockarea", dockWidgetArea(dockWidget_Settings));
    // settings.setValue("docked", dockWidget_Settings->isFloating());
    // settings.setValue("hidden", dockWidget_Settings->isHidden());
    settings.setValue( "size", dockWidget_Settings->topLevelWidget()->size() );
    settings.setValue( "pos", dockWidget_Settings->pos() );
    settings.endGroup();

    // Plots Dock
    settings.beginGroup( "PlotsDock" );
    // settings.setValue("dockarea", dockWidgetArea(dockWidget_Plots));
    // settings.setValue("docked", dockWidget_Plots->isFloating());
    // settings.setValue("hidden", dockWidget_Plots->isHidden());
    settings.setValue( "size", dockWidget_Plots->topLevelWidget()->size() );
    settings.setValue( "pos", dockWidget_Plots->pos() );
    settings.endGroup();

    // Debug Dock
    settings.beginGroup( "DebugDock" );
    // settings.setValue("dockarea", dockWidgetArea(dockWidget_Debug));
    // settings.setValue("docked", dockWidget_Debug->isFloating());
    // settings.setValue("hidden", dockWidget_Debug->isHidden());
    settings.setValue( "size", dockWidget_Debug->topLevelWidget()->size() );
    settings.setValue( "pos", dockWidget_Debug->pos() );
    settings.endGroup();
}

void MainWindow::closeEvent( QCloseEvent * event )
{
    this->controlWidget->stop_all();

    writeSettings();
    this->spinWidget->close();
    this->controlWidget->close();
    this->infoWidget->close();

    event->accept();
}