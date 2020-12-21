#pragma once
#ifndef SPIRIT_CONTROLWIDGET_HPP
#define SPIRIT_CONTROLWIDGET_HPP

#include "ui_ControlWidget.h"

#include "SettingsWidget.hpp"
#include "SpinWidget.hpp"

#include <QFutureWatcher>
#include <QWidget>

#include <memory>
#include <thread>

struct State;

/*
    Converts a QString to an std::string.
    This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std( QString qs );

class ControlWidget : public QWidget, private Ui::ControlWidget
{
    Q_OBJECT

public:
    ControlWidget( std::shared_ptr<State> state, SpinWidget * spinWidget, SettingsWidget * settingsWidget );
    void updateData();
    void cycleMethod();
    void cycleSolver();
    std::string methodName();
    std::string solverName();

public slots:
    void play_pause();
    void stop_all();
    void stop_current();
    void next_image();
    void prev_image();
    void jump_to_image();
    void cut_image();
    void paste_image( std::string where = "current" );
    void delete_image();
    void next_mode();
    void prev_mode();
    void jump_to_mode();
    void calculate();
    void apply_mode();
    void calculate_disable_widget();
    void calculate_enable_widget();
    void ema_buttons_show();
    void ema_buttons_hide();

private slots:
    void resetPressed();
    void xPressed();
    void yPressed();
    void zPressed();
    void save_EPressed();
    void set_solver_enabled();

private:
    void updateOthers();
    void readSettings();
    void writeSettings();
    // State
    std::shared_ptr<State> state;
    // Spin Widget
    SpinWidget * spinWidget;
    // Settings Widget
    SettingsWidget * settingsWidget;

    // Method vectors
    std::vector<std::thread> threads_image;
    std::thread thread_chain;

    // Temporary string storage
    std::string s_method;
    std::string s_solver;

    // Image index of last update
    int idx_image_last;

    // Future watcher for calculation of eigenmodes
    QFutureWatcher<void> watcher;

    // ...
    void save_Energies();

protected:
    void closeEvent( QCloseEvent * event );
};

#endif
