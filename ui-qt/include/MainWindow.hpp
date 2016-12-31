#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "ControlWidget.hpp"
#include "SettingsWidget.hpp"
#include "PlotsWidget.hpp"
#include "DebugWidget.hpp"

#include "ui_MainWindow.h"

// Forward declarations
class QAction;
class QMenu;
class QPlainTextEdit;
struct State;

std::string string_q2std(QString qs);

class MainWindow : public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT

public:
    MainWindow(std::shared_ptr<State> state);

protected:
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;

private slots:
	void keyPressEvent(QKeyEvent *ev) override;
	void view_toggleDebug();
	void view_togglePlots();
	void view_toggleSettings();
	void view_toggle_fullscreen_spins();
    void about();
	void keyBindings();
	void load_Configuration();
	void save_Spin_Configuration();
	void load_Spin_Configuration();
	void save_SpinChain_Configuration();
	void load_SpinChain_Configuration();
	void save_Energies();
	void return_focus();

private:
    /*void createWidgets(Spin_System * s);
    void createActions();
    void createMenus();
    void createToolBars();*/
    void readSettings();
    void writeSettings();
	void createStatusBar();
	void updateStatusBar();

	// State
	std::shared_ptr<State> state;
	// Widgets
	SpinWidget *spinWidget;
	ControlWidget * controlWidget;
	SettingsWidget * settingsWidget;
	PlotsWidget * plotsWidget;
	DebugWidget * debugWidget;
	
	// Update Timers for all Widgets
	QTimer * m_timer_spins;
	QTimer * m_timer_control;
	QTimer * m_timer_plots;
	QTimer * m_timer_debug;
	QTimer * m_timer;

	// Status Bar labels
	QLabel * m_Spacer_1;
	QLabel * m_Spacer_2;
	QLabel * m_Spacer_3;
	QLabel * m_Spacer_4;
	QLabel * m_Label_E;
	QLabel * m_Label_Mz;
	QLabel * m_Label_Torque;
	QLabel * m_Label_NOC;
	QLabel * m_Label_NOI;
	QLabel * m_Label_NOS;
	QLabel * m_Label_FPS;
	std::vector<QLabel*> m_Labels_IPS;

	// Fullscreen state
	bool   fullscreen_spins;
	bool   pre_fullscreen_settings_hidden;
	QSize  pre_fullscreen_settings_size;
	QPoint pre_fullscreen_settings_pos;
	bool   pre_fullscreen_plots_hidden;
	QSize  pre_fullscreen_plots_size;
	QPoint pre_fullscreen_plots_pos;
	bool   pre_fullscreen_debug_hidden;
	QSize  pre_fullscreen_debug_size;
	QPoint pre_fullscreen_debug_pos;
};

#endif
