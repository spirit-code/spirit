// #include <QtWidgets>

#include "DebugWidget.h"
#include "Interface_Log.h"

DebugWidget::DebugWidget(std::shared_ptr<State> state)
{
	this->state = state;
    
	// Setup User Interface
    this->setupUi(this);

	// Load variables
	this->comboBox_ShowLevel->setCurrentIndex(3); // (int)Log.print_level);
	this->plainTextEdit->setPlainText("");
	this->n_log_entries = 0;

	// Connect Signals
	connect(this->comboBox_ShowLevel, static_cast<void (QComboBox::*)(const QString &)>(&QComboBox::currentIndexChanged), this, &DebugWidget::LoadFromLog);
	connect(this->pushButton_All, &QPushButton::clicked, this, &DebugWidget::AllPressed);
	connect(this->checkBox_IO, &QCheckBox::stateChanged, this, &DebugWidget::LoadFromLog);
	connect(this->checkBox_GUI, &QCheckBox::stateChanged, this, &DebugWidget::LoadFromLog);
	connect(this->checkBox_LLG, &QCheckBox::stateChanged, this, &DebugWidget::LoadFromLog);
	connect(this->checkBox_GNEB, &QCheckBox::stateChanged, this, &DebugWidget::LoadFromLog);
	connect(this->checkBox_MMF, &QCheckBox::stateChanged, this, &DebugWidget::LoadFromLog);

	this->plainTextEdit->setFont(QFont("Courier"));
	this->plainTextEdit->ensureCursorVisible();

	// Pull text from Log
	this->UpdateFromLog();
}

void DebugWidget::update()
{
	// Update the list of log entries
	if (n_log_entries < Log_Get_N_Entries(state.get()))
	{
		this->UpdateFromLog();
	}
}

void DebugWidget::UpdateFromLog()
{
	// Load all new Log messages and apply filters
	auto entries = Log_Get_Entries(state.get());
	auto n_old_entries = this->n_log_entries;
	this->n_log_entries = entries.size();
	for (int i = n_old_entries; i < this->n_log_entries; ++i)
	{
		if ((int)entries[i].level <= this->comboBox_ShowLevel->currentIndex())
		{
			if ((entries[i].sender == Utility::Log_Sender::ALL) ||
				(this->checkBox_API->isChecked() && (entries[i].sender == Utility::Log_Sender::API)) ||
				(this->checkBox_IO->isChecked() && (entries[i].sender == Utility::Log_Sender::IO)) ||
				(this->checkBox_GUI->isChecked() && (entries[i].sender == Utility::Log_Sender::UI)) ||
				(this->checkBox_LLG->isChecked() && (entries[i].sender == Utility::Log_Sender::LLG)) ||
				(this->checkBox_GNEB->isChecked() && (entries[i].sender == Utility::Log_Sender::GNEB)) ||
				(this->checkBox_MMF->isChecked() && (entries[i].sender == Utility::Log_Sender::MMF)))
			{
				this->plainTextEdit->appendPlainText(QString::fromLatin1(LogEntryToString(entries[i]).c_str()));
			}
		}
	}
}

void DebugWidget::LoadFromLog()
{
	// Reload all Log messages
	this->plainTextEdit->setPlainText("");
	this->n_log_entries = 0;
	this->UpdateFromLog();
}

void DebugWidget::AllPressed()
{
	// All checkboxes should be checked
	this->checkBox_IO->setChecked(true);
	this->checkBox_GUI->setChecked(true);
	this->checkBox_LLG->setChecked(true);
	this->checkBox_GNEB->setChecked(true);
	this->checkBox_MMF->setChecked(true);
	// Reload the Log
	this->LoadFromLog();
}