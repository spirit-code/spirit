#include <QtWidgets>

#include "ControlWidget.hpp"

#include "Spirit/System.h"
#include "Spirit/Chain.h"
#include "Spirit/Collection.h"
#include "Spirit/Simulation.h"
#include "Spirit/IO.h"
#include "Spirit/Log.h"


std::string string_q2std(QString qs)
{
	auto bytearray = qs.toLatin1();
	const char *c_fileName = bytearray.data();
	return std::string(c_fileName);
}

ControlWidget::ControlWidget(std::shared_ptr<State> state, SpinWidget *spinWidget, SettingsWidget *settingsWidget)
{
	this->state = state;
	this->spinWidget = spinWidget;
	this->settingsWidget = settingsWidget;
    
	// Create threads
	threads_llg = std::vector<std::thread>(Chain_Get_NOI(this->state.get()));
	threads_gneb = std::vector<std::thread>(Collection_Get_NOC(this->state.get()));
	//threads_mmf

	// Setup User Interface
    this->setupUi(this);

	// Buttons
	connect(this->lineEdit_Save_E, SIGNAL(returnPressed()), this, SLOT(save_EPressed()));
	connect(this->pushButton_Save_E, SIGNAL(clicked()), this, SLOT(save_EPressed()));
	connect(this->pushButton_StopAll, SIGNAL(clicked()), this, SLOT(stop_all()));
	connect(this->pushButton_PlayPause, SIGNAL(clicked()), this, SLOT(play_pause()));
	connect(this->pushButton_PreviousImage, SIGNAL(clicked()), this, SLOT(prev_image()));
	connect(this->pushButton_NextImage, SIGNAL(clicked()), this, SLOT(next_image()));
	connect(this->lineEdit_ImageNumber, SIGNAL(returnPressed()), this, SLOT(jump_to_image()));
    connect(this->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetPressed()));
    connect(this->pushButton_X, SIGNAL(clicked()), this, SLOT(xPressed()));
    connect(this->pushButton_Y, SIGNAL(clicked()), this, SLOT(yPressed()));
    connect(this->pushButton_Z, SIGNAL(clicked()), this, SLOT(zPressed()));
	connect(this->comboBox_Method, SIGNAL(currentIndexChanged(int)), this, SLOT(set_solver_enabled()));

	// Image number
	// We use a regular expression (regex) to filter the input into the lineEdits
	QRegularExpression re("[\\d]*");
	QRegularExpressionValidator *number_validator = new QRegularExpressionValidator(re);
	this->lineEdit_ImageNumber->setValidator(number_validator);
	this->lineEdit_ImageNumber->setText(QString::number(1));

	// Read persistent settings
	this->readSettings();
}

void ControlWidget::updateData()
{
	// Check for running simulations - update Play/Pause Button
	if ( Simulation_Running_Collection(state.get()) ||
		 Simulation_Running_Chain(state.get())      ||
		 Simulation_Running_Image(state.get())      )
	{
		this->pushButton_PlayPause->setText("Pause");
		this->spinWidget->updateData();
	}
	else
    {
        if( this->pushButton_PlayPause->text() == "Pause" )
            this->spinWidget->updateData();
		this->pushButton_PlayPause->setText("Play");
    }

	// Update Image number
	this->lineEdit_ImageNumber->setText(QString::number(System_Get_Index(state.get())+1));
	// Update NOI counter
	this->label_NOI->setText("/ " + QString::number(Chain_Get_NOI(state.get())));

	// Update thread arrays
	if (Chain_Get_NOI(state.get()) > (int)threads_llg.size())
	{
		for (int i=threads_llg.size(); i < Chain_Get_NOI(state.get()); ++i)
			this->threads_llg.push_back(std::thread());
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
	int idx = this->comboBox_Method->currentIndex();
	int idx_max = this->comboBox_Method->count();
	this->comboBox_Method->setCurrentIndex((idx + 1) % idx_max);
}

void ControlWidget::cycleSolver()
{
	int idx = this->comboBox_Solver->currentIndex();
	int idx_max = this->comboBox_Solver->count();
	this->comboBox_Solver->setCurrentIndex((idx + 1) % idx_max);
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
	
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "Button: playpause");

	Chain_Update_Data(this->state.get());

	auto qs_method = this->comboBox_Method->currentText();
	auto qs_solver = this->comboBox_Solver->currentText();
	
	this->s_method = string_q2std(qs_method);
	this->s_solver = string_q2std(qs_solver);
	
	auto c_method = s_method.c_str();
	auto c_solver = s_solver.c_str();

	if ( Simulation_Running_Image(this->state.get()) ||
		 Simulation_Running_Chain(this->state.get()) ||
		 Simulation_Running_Collection(this->state.get()) )
	{
		// Running, so we stop it
		Simulation_PlayPause(this->state.get(), c_method, c_solver);
		// Join the thread of the stopped simulation
		if (threads_llg[System_Get_Index(state.get())].joinable()) threads_llg[System_Get_Index(state.get())].join();
		else if (threads_gneb[Chain_Get_Index(state.get())].joinable()) threads_gneb[Chain_Get_Index(state.get())].join();
		else if (thread_mmf.joinable()) thread_mmf.join();
		// New button text
		this->pushButton_PlayPause->setText("Play");
	}
	else
	{
		// Not running, so we start it
		if (this->s_method == "LLG")
		{
			int idx = System_Get_Index(state.get());
			if (threads_llg[idx].joinable()) threads_llg[System_Get_Index(state.get())].join();
			this->threads_llg[System_Get_Index(state.get())] =
				std::thread(&Simulation_PlayPause, this->state.get(), c_method, c_solver, -1, -1, -1, -1);
		}
		else if (this->s_method == "MC")
		{
			int idx = System_Get_Index(state.get());
			if (threads_llg[idx].joinable()) threads_llg[System_Get_Index(state.get())].join();
			this->threads_llg[System_Get_Index(state.get())] =
				std::thread(&Simulation_PlayPause, this->state.get(), c_method, c_solver, -1, -1, -1, -1);
		}
		else if (this->s_method == "GNEB")
		{
			if (threads_gneb[Chain_Get_Index(state.get())].joinable()) threads_gneb[Chain_Get_Index(state.get())].join();
			this->threads_gneb[Chain_Get_Index(state.get())] =
				std::thread(&Simulation_PlayPause, this->state.get(), c_method, c_solver, -1, -1, -1, -1);
		}
		else if (this->s_method == "MMF")
		{
			if (thread_mmf.joinable()) thread_mmf.join();
			this->thread_mmf =
				std::thread(&Simulation_PlayPause, this->state.get(), c_method, c_solver, -1, -1, -1, -1);
		}
		// New button text
		this->pushButton_PlayPause->setText("Pause");
	}
}

void ControlWidget::stop_all()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "Button: stopall");
	
	Simulation_Stop_All(state.get());

	for (unsigned int i=0; i<threads_llg.size(); ++i)
	{
		if (threads_llg[i].joinable()) threads_llg[i].join();
	}
	for (unsigned int i=0; i<threads_gneb.size(); ++i)
	{
		if (threads_gneb[i].joinable()) threads_gneb[i].join();
	}
	if (thread_mmf.joinable()) thread_mmf.join();

	this->pushButton_PlayPause->setText("Play");
	// this->createStatusBar();
}

void ControlWidget::stop_current()
{
	Log_Send(state.get(), Log_Level_Debug, Log_Sender_UI, "Button: stopall");
	
	if ( Simulation_Running_Image(this->state.get()) ||
		 Simulation_Running_Chain(this->state.get()) ||
		 Simulation_Running_Collection(this->state.get()) )
	{
		// Running, so we stop it
		Simulation_PlayPause(this->state.get(), "", "");
		// Join the thread of the stopped simulation
		if (threads_llg[System_Get_Index(state.get())].joinable()) threads_llg[System_Get_Index(state.get())].join();
		else if (threads_gneb[Chain_Get_Index(state.get())].joinable()) threads_gneb[Chain_Get_Index(state.get())].join();
		else if (thread_mmf.joinable()) thread_mmf.join();
	}

	this->pushButton_PlayPause->setText("Play");
}



void ControlWidget::next_image()
{
	if (System_Get_Index(state.get()) < Chain_Get_NOI(this->state.get())-1)
	{
		// Change active image
		Chain_next_Image(this->state.get());

		// Update
		this->updateData();
		this->updateOthers();
	}
}

void ControlWidget::prev_image()
{
	// this->return_focus();
	if (System_Get_Index(state.get()) > 0)
	{
		// Change active image!
		Chain_prev_Image(this->state.get());
		
		// Update
		this->updateData();
		this->updateOthers();
	}
}

void ControlWidget::jump_to_image()
{
	// Change active image
	int idx = this->lineEdit_ImageNumber->text().toInt()-1;
	Chain_Jump_To_Image(this->state.get(), idx);
	
	// Update
	this->updateData();
	this->updateOthers();
}

void ControlWidget::cut_image()
{
	if (Chain_Get_NOI(state.get()) > 1)
	{
		this->stop_current();

		Chain_Image_to_Clipboard(state.get());

		int idx = System_Get_Index(state.get());
		if (Chain_Delete_Image(state.get(), idx))
		{
			// Make the llg_threads vector smaller
			if (this->threads_llg[idx].joinable()) this->threads_llg[idx].join();
			this->threads_llg.erase(threads_llg.begin() + idx);
		}
	}

	// Update
	this->updateData();
	this->updateOthers();
}

void ControlWidget::paste_image(std::string where)
{
	if (where == "current")
	{
		// Paste a Spin System into current System
		this->stop_current();
		Chain_Replace_Image(state.get());
	}
	else if (where == "left")
	{
		int idx = System_Get_Index(state.get());
		// Insert Image
		Chain_Insert_Image_Before(state.get());
		// Make the llg_threads vector larger
		this->threads_llg.insert(threads_llg.begin()+idx, std::thread());
		// Switch to the inserted image
		Chain_prev_Image(this->state.get());
	}
	else if (where == "right")
	{
		int idx = System_Get_Index(state.get());
		// Insert Image
		Chain_Insert_Image_After(state.get());
		// Make the llg_threads vector larger
		this->threads_llg.insert(threads_llg.begin()+idx+1, std::thread());
		// Switch to the inserted image
		Chain_next_Image(this->state.get());
	}

	// Update
	this->updateData();
	this->updateOthers();
}

void ControlWidget::delete_image()
{
	if (Chain_Get_NOI(state.get()) > 1)
	{
		this->stop_current();

		int idx = System_Get_Index(state.get());
		if (Chain_Delete_Image(state.get()))
		{
			// Make the llg_threads vector smaller
			if (this->threads_llg[idx].joinable()) this->threads_llg[idx].join();
			this->threads_llg.erase(threads_llg.begin() + idx);
		}

		Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("Deleted image " + std::to_string(System_Get_Index(state.get()))).c_str());
	}
	
	// Update
	this->updateData();
	this->updateOthers();
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
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save Energies"), "./output", tr("Text (*.txt)"));
	if (!fileName.isEmpty()) {
		auto file = string_q2std(fileName);
		IO_Chain_Write_Energies(this->state.get(), file.c_str());
	}
}

void ControlWidget::set_solver_enabled()
{
	if (this->comboBox_Method->currentText() == "MC")
		this->comboBox_Solver->setEnabled(false);
	else
		this->comboBox_Solver->setEnabled(true);
}

void ControlWidget::save_EPressed()
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
	IO_Image_Write_Energy_per_Spin(this->state.get(), fullNameSpins.c_str());
	IO_Chain_Write_Energies(this->state.get(), fullName.c_str());
	IO_Chain_Write_Energies_Interpolated(this->state.get(), fullNameInterpolated.c_str());

	// Update File name in LineEdit if it fits the schema
	size_t found = fileName.find("Energies");
	if (found != std::string::npos) {
		int a = std::stoi(fileName.substr(found+9, 3)) + 1;
		char newName[20];
		snprintf(newName, 20, "Energies_%03i.txt", a);
		lineEdit_Save_E->setText(newName);
	}
}

void ControlWidget::readSettings()
{
	QSettings settings("Spirit Code", "Spirit");

	// Method and Solver
	settings.beginGroup("ControlWidget");
	this->comboBox_Method->setCurrentIndex(settings.value("Method").toInt());
	this->comboBox_Solver->setCurrentIndex(settings.value("Solver").toInt());
	settings.endGroup();
}

void ControlWidget::writeSettings()
{
	QSettings settings("Spirit Code", "Spirit");

	// Method and Solver
	settings.beginGroup("ControlWidget");
	settings.setValue("Method", this->comboBox_Method->currentIndex());
	settings.setValue("Solver", this->comboBox_Solver->currentIndex());
	settings.endGroup();
}


void ControlWidget::closeEvent(QCloseEvent *event)
{
	writeSettings();
	event->accept();
}