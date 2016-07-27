#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <memory>
#include <map>

#include "Spin_Widget.h"
#include "SettingsWidget.h"
#include "PlotsWidget.h"
#include "DebugWidget.h"
#include "Spin_System.h"
#include "Spin_System_Chain.h"

#include "Solver_LLG.h"
#include "Method_GNEB.h"

#include "Interface_State.h"

class QAction;
class QMenu;
class QPlainTextEdit;

#include "ui_MainWindow.h"

class MainWindow : public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT

public:
    MainWindow(std::shared_ptr<State> state);
	//bool eventFilter(QObject *object, QEvent *event);

protected:
    void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;
    //bool eventFilter(QObject *obj, QEvent *event);

private slots:
    //void newFile();
    //void open();
    //bool save();
    //bool saveAs();
	void keyPressEvent(QKeyEvent *ev) override;
	void playpausePressed();
	void stopallPressed();
	void previousImagePressed();
	void nextImagePressed();
    void resetPressed();
    void xPressed();
    void yPressed();
    void zPressed();
	void view_toggleDebug();
	void view_togglePlots();
	void view_toggleSettings();
    void about();
	void keyBindings();
	void load_Configuration();
	void save_Spin_Configuration();
	void load_Spin_Configuration();
	void save_SpinChain_Configuration();
	void load_SpinChain_Configuration();
	void save_Energies();
	void save_EPressed();
	void return_focus();
	//Interactions
	
	//void documentWasModified();

private:
    /*void createWidgets(Spin_System * s);
    void createActions();
    void createMenus();
    void createToolBars();*/
    void readSettings();
    void writeSettings();
	void createStatusBar();
	void updateStatusBar();
	std::vector<double> getIterationsPerSecond();
    /*bool maybeSave();
    void loadFile(const QString &fileName);
    bool saveFile(const QString &fileName);
    void setCurrentFile(const QString &fileName);
    QString strippedName(const QString &fullFileName);*/
    
    //SpinWidget *spins;
	std::shared_ptr<State> state;
    Spin_Widget *spinWidget;
	SettingsWidget * settingsWidget;
	PlotsWidget * plotsWidget;
	DebugWidget * debugWidget;
    /*QPlainTextEdit *textEdit;
    QString curFile;*/

	// Update Timers for all Widgets
	QTimer * m_timer_spins;
	QTimer * m_timer_plots;
	QTimer * m_timer_debug;
	QTimer * m_timer;

	// Status Bar labels
	QLabel * m_Label_NOI;
	QLabel * m_Label_NOS;
	QLabel * m_Label_FPS;
	std::vector<QLabel*> m_Labels_IPS;

	// Image/Chain - Solver maps
	std::map<std::shared_ptr<Data::Spin_System>, Engine::Solver_LLG*> llg_solvers;
	std::map<std::shared_ptr<Data::Spin_System_Chain>, Engine::Method_GNEB*> gneb_solvers;

};

#endif