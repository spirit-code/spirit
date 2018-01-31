#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "InfoWidget.hpp"
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
	void takeScreenshot();
	void edit_cut();
	void edit_copy();
	void edit_paste();
	void edit_insert_right();
	void edit_insert_left();
	void edit_delete();
	void control_random();
	void control_insertconfiguration();
	void control_playpause();
	void control_cycle_method();
	void control_cycle_solver();
	void view_toggleDebug();
	void view_toggleGeometry();
	void view_togglePlots();
	void view_toggleSettings();
	void view_regular_mode();
	void view_isosurface_mode();
	void view_slab_x();
	void view_slab_y();
	void view_slab_z();
	void view_cycle_camera();
	void view_toggle_spins_only();
	void view_toggle_fullscreen();
	void toggleSpinWidget();
	void toggleInfoWidget();
	void view_toggleDragMode();
    void about();
	void keyBindings();
	void load_Configuration();
	void save_Configuration();
	void save_Spin_Configuration();
	void load_Spin_Configuration();
	void save_SpinChain_Configuration();
	void load_SpinChain_Configuration();
	void save_System_Energy_Spins();
	void save_Chain_Energies();
	void save_Chain_Energies_Interpolated();
	void return_focus();
	void updateMenuBar();

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
	InfoWidget *infoWidget;
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
	QLabel * m_Spacer_5;
	QLabel * m_Label_E;
	QLabel * m_Label_Mz;
	QLabel * m_Label_Torque;
	QLabel * m_Label_NOC;
	QLabel * m_Label_NOI;
	QLabel * m_Label_NOS;
	QLabel * m_Label_Dims;
	QLabel * m_Label_FPS;
	std::vector<QLabel*> m_Labels_IPS;

	// Fullscreen state
	bool   view_spins_only;
	bool   view_fullscreen;
	bool   m_spinWidgetActive;
	bool   m_InfoWidgetActive;
	bool   pre_spins_only_settings_hidden;
	QSize  pre_spins_only_settings_size;
	QPoint pre_spins_only_settings_pos;
	bool   pre_spins_only_plots_hidden;
	QSize  pre_spins_only_plots_size;
	QPoint pre_spins_only_plots_pos;
	bool   pre_spins_only_debug_hidden;
	QSize  pre_spins_only_debug_size;
	QPoint pre_spins_only_debug_pos;

	// Screenshot numbering
	int n_screenshots;
};

#endif
