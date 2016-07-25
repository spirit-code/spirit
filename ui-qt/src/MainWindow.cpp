#include <QtWidgets>

#include "MainWindow.h"
#include "PlotWidget.h"

#include "Vectormath.h"
#include "Configurations.h"
#include "Optimizer.h"
#include "IO.h"

#include "Logging.h"
#include "Timing.h"
#include "Threading.h"

#include <thread>

MainWindow::MainWindow(std::shared_ptr<State> state)
{
	// State
	this->state = state;
	// Widgets
	this->spinWidget = new Spin_Widget(this->state);
	//this->spinWidgetGL = new Spin_Widget_GL(s);
	this->settingsWidget = new SettingsWidget(this->state);
	this->plotsWidget = new PlotsWidget(this->state);
	this->debugWidget = new DebugWidget(this->state);

	//this->setFocus(Qt::StrongFocus);
	this->setFocusPolicy(Qt::StrongFocus);

	// Fix text size on OSX
    #ifdef Q_OS_MAC
        this->setStyleSheet("QWidget{font-size:10pt}");
    #else
        this->setStyleSheet("QWidget{font-size:8pt}");
    #endif
    
	// Setup User Interface
    this->setupUi(this);

	// Tabify DockWidgets for Plots and Debug
	this->tabifyDockWidget(this->dockWidget_Plots, this->dockWidget_Debug);
	this->dockWidget_Plots->raise();
	this->dockWidget_Debug->hide();

	// Read Window settings of last session
	readSettings();

	// Add Widgets to UIs grids
	this->gridLayout->addWidget(this->spinWidget, 0, 0, 1, 1);
	this->dockWidget_Settings->setWidget(this->settingsWidget);
	this->dockWidget_Plots->setWidget(this->plotsWidget);
	this->dockWidget_Debug->setWidget(this->debugWidget);

	// Read Iterate State form Spin System
	if (state->active_image->iteration_allowed == false)
	{
		this->pushButton_PlayPause->setText("Play");
	}
	else
	{
		this->pushButton_PlayPause->setText("Pause");
		//std::thread(Engine::SIB::Iterate, s).detach();
	}

	/*
    // Create Widgets
    createWidgets(s);
    
    // Create Stuff
    createActions();
    //createMenus();    // these fail for some reason... maybe add resource stuff later on
    //createToolBars(); // these fail for some reason... maybe add resource stuff later on

    connect(textEdit->document(), SIGNAL(contentsChanged()), this, SLOT(documentWasModified()));

    setCurrentFile("");
    setUnifiedTitleAndToolBarOnMac(true);//*/
    

	// Set up Update Timers
	m_timer = new QTimer(this);
	//m_timer_plots = new QTimer(this);
	//m_timer_spins = new QTimer(this);


	// Buttons
	connect(this->lineEdit_Save_E, SIGNAL(returnPressed()), this, SLOT(save_EPressed()));
	connect(this->pushButton_Save_E, SIGNAL(clicked()), this, SLOT(save_EPressed()));
	connect(this->pushButton_StopAll, SIGNAL(clicked()), this, SLOT(stopallPressed()));
	//connect(this->pushButton_StopAll, SIGNAL(clicked()), this, SLOT(createStatusBar()));
	connect(this->pushButton_PlayPause, SIGNAL(clicked()), this, SLOT(playpausePressed()));
	//connect(this->pushButton_PlayPause, SIGNAL(clicked()), this, SLOT(createStatusBar()));
	connect(this->pushButton_PreviousImage, SIGNAL(clicked()), this, SLOT(previousImagePressed()));
	connect(this->pushButton_NextImage, SIGNAL(clicked()), this, SLOT(nextImagePressed()));
    connect(this->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetPressed()));
    connect(this->pushButton_X, SIGNAL(clicked()), this, SLOT(xPressed()));
    connect(this->pushButton_Y, SIGNAL(clicked()), this, SLOT(yPressed()));
    connect(this->pushButton_Z, SIGNAL(clicked()), this, SLOT(zPressed()));


	// Image number
	// We use a regular expression (regex) to filter the input into the lineEdits
	QRegularExpression re("[\\d]*");
	QRegularExpressionValidator *number_validator = new QRegularExpressionValidator(re);
	this->lineEdit_ImageNumber->setValidator(number_validator);
	this->lineEdit_ImageNumber->setText(QString::number(1));

	// File Menu
	connect(this->actionLoad_Configuration, SIGNAL(triggered()), this, SLOT(load_Configuration()));
	connect(this->actionLoad_Spin_Configuration, SIGNAL(triggered()), this, SLOT(load_Spin_Configuration()));
	connect(this->actionLoad_SpinChain_Configuration, SIGNAL(triggered()), this, SLOT(load_SpinChain_Configuration()));
	connect(this->actionSave_Energies, SIGNAL(triggered()), this, SLOT(save_Energies()));
	connect(this->action_Save_Spin_Configuration, SIGNAL(triggered()), SLOT(save_Spin_Configuration()));
	connect(this->actionSave_SpinChain_Configuration, SIGNAL(triggered()), this, SLOT(save_SpinChain_Configuration()));

	// View Menu
	connect(this->actionShow_Settings, SIGNAL(triggered()), this, SLOT(view_toggleSettings()));
	connect(this->actionShow_Plots, SIGNAL(triggered()), this, SLOT(view_togglePlots()));
	connect(this->actionShow_Debug, SIGNAL(triggered()), this, SLOT(view_toggleDebug()));

	// Help Menu
	connect(this->actionKey_Bindings, SIGNAL(triggered()), this, SLOT(keyBindings()));	
	connect(this->actionAbout_this_Application, SIGNAL(triggered()), this, SLOT(about()));

	// Status Bar
	//		FPS
	this->m_Label_FPS = new QLabel;
	this->m_Label_FPS->setText("FPS: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(m_Label_FPS);
	//		NOS
	this->m_Label_NOS = new QLabel;
	this->m_Label_NOS->setText("NOS: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOS);
	//		NOI
	this->m_Label_NOI = new QLabel;
	this->m_Label_NOI->setText("NOI: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOI);
	//		Initialisations & connect
	this->createStatusBar();
	connect(m_timer, &QTimer::timeout, this, &MainWindow::updateStatusBar);

	// Plots Widget
	//connect(m_timer_plots, &QTimer::timeout, this->plotsWidget->energyPlot, &PlotWidget::update);	// this currently resets the user's interaction (movement, zoom)

	// Spins Widget
	//connect(m_timer_spins, &QTimer::timeout, this->spinWidget, &Spin_Widget::update);
	

	// Event Filter
	//this->installEventFilter(this);

	// Start Timers
	m_timer->start(200);
	//m_timer_plots->start(100);
	//m_timer_spins->start(100);
	//m_timer_debug->start(100);



	Ui::MainWindow::statusBar->showMessage(tr("Ready"), 5000);
}


void MainWindow::keyPressEvent(QKeyEvent *k)
{
	// Key Sequences
	if (k->matches(QKeySequence::Copy))
	{
		// Copy a Spin System
		image_clipboard = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*state->active_image));
		Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Copied image " + std::to_string(state->idx_active_image) + " to clipboard");
	}
	else if (k->matches(QKeySequence::Cut))
	{
		if (state->noi > 1)
		{
			state->active_image->iteration_allowed = false;
			state->active_chain->iteration_allowed = false;
			if (Utility::Threading::llg_threads[state->active_image].joinable()) {
				Utility::Threading::llg_threads[state->active_image].join();
			}
			if (Utility::Threading::gneb_threads[state->active_chain].joinable()) {
				Utility::Threading::gneb_threads[state->active_chain].join();
			}

			// Cut a Spin System
			image_clipboard = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*state->active_image));

			int idx = state->idx_active_image;
			if (idx > 0) this->previousImagePressed();
			//else this->nextImagePressed();

			// TODO: Should use Interface function here
			this->state->active_chain->Delete_Image(idx);

			Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Cut image " + std::to_string(state->idx_active_image) + " to clipboard");
		}
	}
	else if (k->matches(QKeySequence::Paste))
	{
		// Paste a Spin System

		state->active_image->iteration_allowed = false;
		state->active_chain->iteration_allowed = false;
		if (Utility::Threading::llg_threads[state->active_image].joinable()) {
			Utility::Threading::llg_threads[state->active_image].join();
		}
		if (Utility::Threading::gneb_threads[state->active_chain].joinable()) {
			Utility::Threading::gneb_threads[state->active_chain].join();
		}

		if (image_clipboard.get())
		{
			// TODO: use interface function
			state->active_image = image_clipboard;
			// TODO: should use interface function
			state->active_chain->Replace_Image(state->idx_active_image, image_clipboard);
			Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Pasted image " + std::to_string(state->idx_active_image) + " from clipboard");
		}
		else Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "Tried to paste image " + std::to_string(state->idx_active_image) + " from clipboard but no image was found");
	}

	// Custom Key Sequences
	else if (k->modifiers() & Qt::ControlModifier)
	{
		std::shared_ptr<Data::Spin_System> newImage;
		switch (k->key())
		{
			// CTRL+Left - Paste image to left of current image
			case Qt::Key_Left:
				if (image_clipboard.get())
				{
					state->active_image = image_clipboard;
					// TODO: should use interface function
					this->state->active_chain->Insert_Image_Before(state->idx_active_image, image_clipboard);
					//this->previousImagePressed();
					Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Pasted image before " + std::to_string(state->idx_active_image) + " from clipboard");
				}
				else Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "Tried to paste image before " + std::to_string(state->idx_active_image) + " from clipboard but no image was found");
				break;

			// CTRL+Right - Paste image to right of current image
			case Qt::Key_Right:
				if (image_clipboard.get())
				{
					state->active_image = image_clipboard;
					// TODO: should use interface function
					this->state->active_chain->Insert_Image_After(state->idx_active_image, image_clipboard);
					this->nextImagePressed();
					Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Pasted image after " + std::to_string(state->idx_active_image) + " from clipboard");
				}
				else Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::GUI, "Tried to paste image after " + std::to_string(state->idx_active_image) + " from clipboard but no image was found");
				break;
		}
	}
	
	// Single Keys
	else
	switch (k->key())
	{
		// Escape: try to return focus to MainWindow
		case Qt::Key_Escape:
			this->setFocus();
			break;
		// Up: ...
		case Qt::Key_Up:
			break;
		// Left: switch to image left of current image
		case Qt::Key_Left:
			this->previousImagePressed();
			break;
		// Left: switch to image left of current image
		case Qt::Key_Right:
			this->nextImagePressed();
			break;
		// Down: ...
		case Qt::Key_Down:
			break;
		// Space: Play and Pause
		case Qt::Key_Space:
			this->playpausePressed();
			break;
		// F1: Show key bindings
		case Qt::Key_F1:
			this->keyBindings();
			break;
		// F2: Toggle settings widget
		case Qt::Key_F2:
			this->view_toggleSettings();
			break;
		// F3: Toggle Plots widget
		case Qt::Key_F3:
			this->view_togglePlots();
			break;
		// F2: Toggle debug widget
		case Qt::Key_F4:
			this->view_toggleDebug();
			break;
		// 0: ...
		case Qt::Key_0:
			break;
		// 1: Select tab 1 of settings widget
		case Qt::Key_1:
			this->settingsWidget->SelectTab(0);
			break;
		// 2: Select tab 2 of settings widget 
		case Qt::Key_2:
			this->settingsWidget->SelectTab(1);
			break;
		// 3: Select tab 3 of settings widget
		case Qt::Key_3:
			this->settingsWidget->SelectTab(2);
			break;
		// 4: Select tab 4 of settings widget
		case Qt::Key_4:
			this->settingsWidget->SelectTab(3);
			break;
		// 5: Select tab 5 of settings widget
		case Qt::Key_5:
			this->settingsWidget->SelectTab(4);
			break;
		// Delete: Delete current image
		case Qt::Key_Delete:
			if (state->noi > 1)
			{
				state->active_image->iteration_allowed = false;
				state->active_chain->iteration_allowed = false;
				if (Utility::Threading::llg_threads[state->active_image].joinable()) {
					Utility::Threading::llg_threads[state->active_image].join();
				}
				if (Utility::Threading::gneb_threads[state->active_chain].joinable()) {
					Utility::Threading::gneb_threads[state->active_chain].join();
				}

				int idx = state->idx_active_image;
				if (idx > 0) this->previousImagePressed();
				//else this->nextImagePressed();
				// TODO: should use interface function
				this->state->active_chain->Delete_Image(idx);

				Utility::Log.Send(Utility::Log_Level::INFO, Utility::Log_Sender::GUI, "Deleted image " + std::to_string(state->idx_active_image));
			}
			break;
	}
}


void MainWindow::stopallPressed()
{
	this->return_focus();
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, std::string("Button: stopall"));
	
	state->active_chain->iteration_allowed = false;
	for (unsigned int i = 0; i < this->gneb_solvers.size(); ++i)
	{
		this->gneb_solvers.erase(state->active_chain);
	}

	for (int i = 0; i < state->noi; ++i)
	{
		state->active_chain->images[i]->iteration_allowed = false;
		this->llg_solvers.erase(state->active_chain->images[i]);
	}

	this->pushButton_PlayPause->setText("Play");
	this->createStatusBar();
}

void MainWindow::playpausePressed()
{
	this->return_focus();
	Utility::Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::GUI, std::string("Button: playpause"));

	this->state->active_chain->Update_Data();

	std::shared_ptr<Engine::Optimizer> optim;
	if (this->comboBox_Optimizer->currentText() == "SIB")
	{
		optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
	}
	else if (this->comboBox_Optimizer->currentText() == "Heun")
	{
		optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_Heun());
	}
	else if (this->comboBox_Optimizer->currentText() == "CG")
	{
		optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_CG());
	}
	else if (this->comboBox_Optimizer->currentText() == "QM")
	{
		optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_QM());
	}

	if (this->comboBox_Solver->currentText() == "LLG")
	{
		if (state->active_image->iteration_allowed)
		{
			this->pushButton_PlayPause->setText("Play");
			state->active_image->iteration_allowed = false;
			state->active_chain->iteration_allowed = false;
			this->gneb_solvers.erase(state->active_chain);
			this->llg_solvers.erase(state->active_image);
		}
		else
		{
			this->pushButton_PlayPause->setText("Pause");
			if (Utility::Threading::llg_threads[state->active_image].joinable())
			{
				state->active_image->iteration_allowed = false;
				Utility::Threading::llg_threads[state->active_image].join();
			}
			state->active_image->iteration_allowed = true;
			state->active_chain->iteration_allowed = false;
            auto g = new Engine::Solver_LLG(this->state->active_chain, optim);
			Utility::Threading::llg_threads[state->active_image] = std::thread(&Engine::Solver_LLG::Iterate, g);
			this->llg_solvers[state->active_image] = g;
		}
	}
	else if (this->comboBox_Solver->currentText() == "GNEB")
	{
		if (state->active_chain->iteration_allowed)
		{
			this->pushButton_PlayPause->setText("Play");
			for (int i = 0; i < state->noi; ++i)
			{
				state->active_chain->images[i]->iteration_allowed = false;
				this->llg_solvers.erase(state->active_chain->images[i]);
			}
			state->active_chain->iteration_allowed = false;
			this->gneb_solvers.erase(state->active_chain);
		}
		else
		{
			this->pushButton_PlayPause->setText("Pause");
			for (int i = 0; i < state->noi; ++i)
			{
				state->active_chain->images[i]->iteration_allowed = false;
			}
			if (Utility::Threading::gneb_threads[state->active_chain].joinable()) {
				state->active_chain->iteration_allowed = false;
				Utility::Threading::gneb_threads[state->active_chain].join();
			}
			state->active_chain->iteration_allowed = true;
            auto g = new Engine::Solver_GNEB(this->state->active_chain, optim);
			Utility::Threading::gneb_threads[state->active_chain] = std::thread(&Engine::Solver_GNEB::Iterate, g);
			this->gneb_solvers[state->active_chain] = g;
		}
	}
	else if (this->comboBox_Solver->currentText() == "MMF")
	{
		Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, std::string("MMF selected, but not yet implemented! Not doing anything..."));
	}
	this->createStatusBar();
}


void MainWindow::previousImagePressed()
{
	this->return_focus();
	if (state->idx_active_image > 0)
	{
		// Change active image!
		// TODO: use interface function
		//Chain_prev_Image(this->state);
		state->idx_active_image--;
		state->active_image = state->active_chain->images[state->idx_active_image];
		this->lineEdit_ImageNumber->setText(QString::number(state->idx_active_image+1));
		// Update Play/Pause Button
		if (this->state->active_image->iteration_allowed || this->state->active_chain->iteration_allowed) this->pushButton_PlayPause->setText("Pause");
		else this->pushButton_PlayPause->setText("Play");

		// Update Image-dependent Widgets
		//this->spinWidget->update();
		this->settingsWidget->update();
		this->plotsWidget->update();
		this->debugWidget->update();
	}
}


void MainWindow::nextImagePressed()
{
	this->return_focus();
	if (state->idx_active_image < this->state->noi-1)
	{
		//Chain_next_Image(this->state);
		// Change active image
		state->idx_active_image++;
		state->active_image = state->active_chain->images[state->idx_active_image];
		this->lineEdit_ImageNumber->setText(QString::number(state->idx_active_image+1));
		// Update Play/Pause Button
		if (this->state->active_image->iteration_allowed || this->state->active_chain->iteration_allowed) this->pushButton_PlayPause->setText("Pause");
		else this->pushButton_PlayPause->setText("Play");

		// Update Image-dependent Widgets
		//this->spinWidget->update();
		this->settingsWidget->update();
		this->plotsWidget->update();
		this->debugWidget->update();
	}
}


void MainWindow::resetPressed()
{
    /*this->spinWidget->SetCameraToDefault();
	this->spinWidget->update();*/
}

void MainWindow::xPressed()
{
    /*this->spinWidget->SetCameraToX();
	this->spinWidget->update();*/
}

void MainWindow::yPressed()
{
    /*this->spinWidget->SetCameraToY();
	this->spinWidget->update();*/
}

void MainWindow::zPressed()
{
    /*this->spinWidget->SetCameraToZ();
	this->spinWidget->update();*/
}

void MainWindow::view_toggleDebug()
{
	if (this->dockWidget_Debug->isVisible()) this->dockWidget_Debug->hide();
	else this->dockWidget_Debug->show();
}

void MainWindow::view_togglePlots()
{
	if (this->dockWidget_Plots->isVisible()) this->dockWidget_Plots->hide();
	else this->dockWidget_Plots->show();
}

void MainWindow::view_toggleSettings()
{
	if (this->dockWidget_Settings->isVisible()) this->dockWidget_Settings->hide();
	else this->dockWidget_Settings->show();
}


void MainWindow::createStatusBar()
{
	// Remove previous IPS labels
	for (unsigned int i = 0; i < this->m_Labels_IPS.size(); ++i)
	{
		Ui::MainWindow::statusBar->removeWidget(this->m_Labels_IPS[i]);
	}

	// Get Solvers' IPS
	auto v_ips = this->getIterationsPerSecond();

	// Create IPS Labels and add them to the statusBar
	this->m_Labels_IPS = std::vector<QLabel*>();
	for (unsigned int i = 0; i < v_ips.size(); ++i)
	{
		this->m_Labels_IPS.push_back(new QLabel);
		this->m_Labels_IPS[i]->setText("IPS [-]: -");
		Ui::MainWindow::statusBar->addPermanentWidget(m_Labels_IPS[i]);
	}

	//		FPS
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_FPS);
	this->m_Label_FPS = new QLabel;
	this->m_Label_FPS->setText("FPS: -");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_FPS);

	//		NOS
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_NOS);
	this->m_Label_NOS = new QLabel;
	this->m_Label_NOS->setText(QString::fromLatin1("NOS: ") + QString::number(this->state->nos));
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOS);

	//		NOI
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_NOI);
	this->m_Label_NOI = new QLabel;
	this->m_Label_NOI->setText(QString::fromLatin1("NOI: ") + QString::number(this->state->noi));
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOI);
}


void MainWindow::updateStatusBar()
{
	this->m_Label_FPS->setText(QString::fromLatin1("FPS: ") + QString::number((int)this->spinWidget->getFramesPerSecond()));
	auto v_ips = this->getIterationsPerSecond();
	for (unsigned int i = 0; i < m_Labels_IPS.size() && i < v_ips.size(); ++i)
	{
		this->m_Labels_IPS[i]->setText(QString::fromLatin1("IPS [") + QString::number(i+1) + QString::fromLatin1("]: ") + QString::number((int)v_ips[i]));
	}
}

std::vector<double> MainWindow::getIterationsPerSecond()
{
	std::vector<double> ret;

	if (this->state->active_chain->iteration_allowed)
	{
		for (unsigned int i = 0; i < this->gneb_solvers.size(); ++i)
		{
			ret.push_back(this->gneb_solvers[state->active_chain]->getIterationsPerSecond());
		}
	}
	else
	{
		for (unsigned int i = 0; i < this->llg_solvers.size(); ++i)
		{
			if (state->active_chain->images[i]->iteration_allowed)
			{
				ret.push_back(this->llg_solvers[state->active_chain->images[i]]->getIterationsPerSecond());
			}
		}
	}

	return ret;
}

void MainWindow::about()
{
	QMessageBox::about(this, tr("About JuSpin"),
		QString::fromLatin1("The <b>JuSpin</b> application incorporates intuitive visualisation,<br>"
			"powerful <b>Spin Dynamics</b> and <b>Nudged Elastic Band</b> tools<br>"
			"into a cross-platform user interface.<br>"
			"<br>"
			"Libraries used are<br>"
			"  - VTK 7<br>"
			"  - QT 5.5<br>"
			"<br>"
			"This has been developed by<br>"
			"  - Gideon M�ller (<a href=\"mailto:g.mueller@fz-juelich.de\">g.mueller@fz-juelich.de</a>)<br>"
			"  - Daniel Sch�rhoff (<a href=\"mailto:d.schuerhoff@fz-juelich.de\">d.schuerhoff@fz-juelich.de</a>)<br>"
			"at the Institute for Advanced Simulation 1 of the Forschungszentrum J�lich.<br>"
			"For more information about us, visit the <a href=\"http://www.fz-juelich.de/pgi/pgi-1/DE/Home/home_node.html\">IAS-1 Website</a><br>"
			"<br>"
			"<b>Copyright 2016</b><br>"));
}

void MainWindow::keyBindings()
{
	QMessageBox::about(this, tr("JuSpin UI Key Bindings"),
		QString::fromLatin1("The <b>Key Bindings</b> are as follows:<br>"
			"<br>"
			" - <b>F1</b>:      Show this<br>"
			" - <b>F2</b>:      Toggle Settings<br>"
			" - <b>F3</b>:      Toggle Plots<br>"
			" - <b>F4</b>:      Toggle Debug<br>"
			"<br>"
			" - <b>1-5</b>:     Select Tab in Settings<br>"
			"<br>"
			" - <b>Arrows</b>:  Switch between arrows and chains<br>"
			" - <b>WASD</b>:    Move the camera around (not yet functional)<br>"
			" - <b>Space</b>:   Play/Pause<br>"
			" - <b>Escape</b>:  Try to return focus to main UI (does not always work)<br>"
			"<br>"
			" - <b>Ctrl+X</b>:           Cut   Image<br>"
			" - <b>Ctrl+C</b>:           Copy  Image<br>"
			" - <b>Ctrl+V</b>:           Paste Image at current index<br>"
			" - <b>Ctrl+Left/Right</b>:  Insert left/right of current index<br>"
			" - <b>Del</b>:              Delete image<br>"));
}

void MainWindow::return_focus()
{
	auto childWidgets = this->findChildren<QWidget *>();
	for (int i = 0; i <childWidgets.count(); ++i)
	{
		childWidgets.at(i)->clearFocus();
	}
	/*this->pushButton_PreviousImage->clearFocus();
	this->pushButton_NextImage->clearFocus();
	this->pushButton_Reset->clearFocus();
	this->pushButton_X->clearFocus();
	this->pushButton_Y->clearFocus();
	this->pushButton_Z->clearFocus();*/
}


/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs)
{
	auto bytearray = qs.toLatin1();
	const char *c_fileName = bytearray.data();
	return std::string(c_fileName);
}

void MainWindow::save_Spin_Configuration()
{
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save Spin Configuration"), "./output", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Append_Spin_Configuration(this->state->active_image, 0, string_q2std(fileName));
	}
}
void MainWindow::load_Spin_Configuration()
{
	auto fileName = QFileDialog::getOpenFileName(this, tr("Load Spin Configuration"), "./input", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Read_Spin_Configuration(this->state->active_image, string_q2std(fileName));
	}
}

void MainWindow::save_SpinChain_Configuration()
{
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save SpinChain Configuration"), "./output", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Save_SpinChain_Configuration(this->state->active_chain, string_q2std(fileName));
	}
}

void MainWindow::load_SpinChain_Configuration()
{
	auto fileName = QFileDialog::getOpenFileName(this, tr("Load Spin Configuration"), "./input", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Read_SpinChain_Configuration(this->state->active_chain, string_q2std(fileName));
	}
}


void MainWindow::save_Energies()
{
	this->return_focus();
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save Energies"), "./output", tr("Text (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Save_Energies(*state->active_chain, 0, string_q2std(fileName));
	}
}

void MainWindow::save_EPressed()
{
	std::string fullName = "output/";
	std::string fullNameSpins = "output/";
	std::string fullNameInterpolated = "output/";

	// Get file info
	auto qFileName = lineEdit_Save_E->text();
	QFileInfo fileInfo(qFileName);
	
	// Construct the file names
	std::string fileName = string_q2std(fileInfo.baseName()) + "." + string_q2std(fileInfo.completeSuffix());
	std::string fileNameSpins = string_q2std(fileInfo.baseName()) + "_Spins." + string_q2std(fileInfo.completeSuffix());
	std::string fileNameInterpolated = string_q2std(fileInfo.baseName()) + "_Interpolated." + string_q2std(fileInfo.completeSuffix());

	// File names including path
	fullName.append(fileName);
	fullNameSpins.append(fileNameSpins);
	fullNameInterpolated.append(fileNameInterpolated);

	// Save Energies and Energies_Spins
	Utility::IO::Save_Energies(*state->active_chain, 0, fullName);
	Utility::IO::Save_Energies_Spins(*state->active_chain, fullNameSpins);
	Utility::IO::Save_Energies_Interpolated(*state->active_chain, fullNameInterpolated);

	// Update File name in LineEdit if it fits the schema
	size_t found = fileName.find("Energies");
	if (found != std::string::npos) {
		int a = std::stoi(fileName.substr(found+9, 3)) + 1;
		char newName[20];
		snprintf(newName, 20, "Energies_%03i.txt", a);
		lineEdit_Save_E->setText(newName);
	}
}

void MainWindow::load_Configuration()
{
	int idx_img = state->idx_active_image;
	// Read System
	auto fileName = QFileDialog::getOpenFileName(this, tr("Open Config"), "./input", tr("Config (*.cfg)"));
	if (!fileName.isEmpty()) {
		std::shared_ptr<Data::Spin_System> sys = Utility::IO::Spin_System_from_Config(string_q2std(fileName));
		// Filter for unacceptable differences to other systems in the chain
		bool acceptable = true;
		for (int i = 0; i < state->noi; ++i)
		{
			if (state->active_chain->images[i]->nos != sys->nos) acceptable = false;
			// Currently the SettingsWidget does not support different images being isotropic AND anisotropic at the same time
			if (state->active_chain->images[i]->is_isotropic != sys->is_isotropic) acceptable = false;
		}
		// Set current image
		if (acceptable)
		{
			this->state->active_chain->images[idx_img] = sys;
			Utility::Configurations::Random(*sys);
		}
		else QMessageBox::about(this, tr("About JuSpin"),
			tr("The resulting Spin System would have different NOS\n"
				"or isotropy status than one or more of the other\n"
				"images in the chain!\n"
				"\n"
				"The system has thus not been reset!"));
	}
}

void MainWindow::readSettings()
{
	QSettings settings("QtProject", "Application Example");
	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();
	resize(size);
	move(pos);
}

void MainWindow::writeSettings()
{
	QSettings settings("QtProject", "Application Example");
	settings.setValue("pos", pos());
	settings.setValue("size", size());
}


void MainWindow::closeEvent(QCloseEvent *event)
{
	state->active_chain->iteration_allowed = false;
	if (Utility::Threading::gneb_threads[state->active_chain].joinable()) Utility::Threading::gneb_threads[state->active_chain].join();
	for (int isystem = 0; isystem < (int)state->active_chain->images.size(); ++isystem)
	{
		state->active_chain->images[isystem]->iteration_allowed = false;
		if (Utility::Threading::llg_threads[state->active_chain->images[isystem]].joinable())
		{
			Utility::Threading::llg_threads[state->active_chain->images[isystem]].join();
		}	
	}
	
    //if (maybeSave()) {
        writeSettings();
        event->accept();
    /*} else {
        event->ignore();
    }*/
}